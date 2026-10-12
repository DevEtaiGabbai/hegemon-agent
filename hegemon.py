#!/usr/bin/env python3
"""Hegemon agent: find a public dataset, convert it the way the lab notebook does, publish it to Hegemon.

  hegemon agent                 asks what to look for, then puts the 10 best datasets on one Hegemon page
                                (ask for the same thing again: only what is new in the best 10 is added)
  hegemon agent --more          the same question, but adds the next 10 down the ranking
  hegemon                       search, pick, build, publish (interactive)
  hegemon search "query"
  hegemon build GSE51984        build one accession (options: --platform, --source, --script, --yes, --no-publish)
  hegemon publish BUILD_FOLDER  publish a finished build
  hegemon compare AGENT_FOLDER LAB_FOLDER   compare with a dataset the lab built by hand
  hegemon mode                  choose local, mixed or cloud models (asked the first time you run hegemon agent)
  hegemon check                 check the server setup and the AI model(s)
  hegemon key                   save or replace the OpenAI API key
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hegemon_nb import compare as compare_mod, convert, evidence, native, net, publish, repos, ui  # noqa: E402
from hegemon_nb.config import MODE_TEXT, MODES, VERSION, Config, load_settings, save_settings  # noqa: E402
from hegemon_nb.geo_soft import read_geo_soft  # noqa: E402
from hegemon_nb.llm import LLM, ProviderError, load_key, save_key  # noqa: E402

REASON_TEXT = {
    "SINGLE_CELL": "Single-cell study. The agent has no single-cell blueprint yet (add single-cell-process.ipynb "
                   "to blueprints/), so it is skipped rather than converted wrongly.",
    "NOT_HOSTED": "The study's metadata names a processed data file that the repository does not actually serve.",
    "TOO_FEW_SAMPLES": "The study has fewer samples than --min-samples: too few points for Hegemon's comparisons.",
    "NO_GROUPING_METADATA": "The samples carry no annotation that splits them into groups (only IDs, titles or "
                            "constant fields), so Hegemon would have nothing to compare or plot by.",
    "NO_PROCESSED_DATA": "The study only posted raw reads/images; there is no processed table to convert.",
    "NO_SAMPLE_MATCH": "The data's columns can't be matched one-to-one to the study's samples.",
    "NOT_EXPRESSION": "This is not a gene/probe-by-sample expression table.",
}
AGENT_FAULT = {
    "FILES_TOO_LARGE": "The data exists; the agent's size limits refused it. Raise HEGEMON_MAX_FILE_GB and "
                       "HEGEMON_MAX_TOTAL_GB, or pass --yes, and run it again.",
    "MISSING_FILES": "A file that should exist was not downloaded or not shown to the AI. This is an agent bug, "
                     "not a fact about the dataset; send me the build folder.",
    "AGENT_GAVE_UP": "This agent could not write a working conversion. That is its limit, not proof the dataset "
                     "cannot be converted; send me the build folder.",
}
RANK_INSTRUCTIONS = (
    "You rank public gene-expression datasets for Professor Debashis Sahoo's lab, which loads them into Hegemon to "
    "compare groups of samples and find Boolean relationships between genes. That only works on datasets with MANY "
    "samples whose annotations split them into groups (disease vs control, treatment, tissue, genotype, time, "
    "response). Reply with JSON only: {\"ranked\": [{\"accession\": \"...\", \"why\": \"<= 12 words\"}]} "
    "listing up to 10 datasets from the list, best first. Rank highest: human (Homo sapiens) cohorts relevant to "
    "the query with the most samples and clear comparison groups, unless the query asks for another organism. "
    "Leave out: single-cell or single-nucleus studies (this lab's single-cell pipeline is separate), datasets "
    "with few samples, single-condition studies, cell lines with no comparison, and anything that is not gene "
    "expression. Also give two human gene symbols that best show the query's biology on a scatter plot (for "
    "example FOXP3 and IL2RA for regulatory T cells), as \"genes\": [\"A\", \"B\"]. Use only accessions from "
    "the list.")


def get_llm(cfg, required=True):
    if not cfg.needs_key():
        return LLM(None, cfg)          # local mode: no OpenAI key, no OpenAI calls
    key = load_key(cfg, prompt=required)
    if not key:
        if required:
            raise SystemExit("An OpenAI API key is needed. Run: hegemon key")
        return None
    return LLM(key, cfg)


def finish(work, receipt, status, detail):
    receipt.update(status=status, detail=detail, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    Path(work).mkdir(parents=True, exist_ok=True)
    (Path(work) / "build.json").write_text(json.dumps(receipt, indent=2))
    ui.fail(status)
    if status in REASON_TEXT:
        ui.info(REASON_TEXT[status] + " (reason given by the AI, from the files)")
    if status in AGENT_FAULT:
        ui.info(AGENT_FAULT[status])
    ui.info(detail)
    ui.info("Everything from this run is in " + str(work))
    return {"status": status, "detail": detail, "work": str(work), "receipt": receipt, "code": 2}


def rank(cfg, llm, query, results, limit=10):
    if llm is None or len(results) < 2:
        return [(d, "") for d in results[:limit]]
    listing = "\n".join("{}\t{}\t{}\t{} samples\t{}\t{}".format(
        d.accession, d.kind, d.organism, d.n_samples or "?", d.title[:160], d.summary[:300].replace("\n", " "))
        for d in results[:max(60, limit)])
    try:
        reply = llm.ask(RANK_INSTRUCTIONS.replace("up to 10", "up to {}".format(limit)),
                        "Query: {}\n\nDatasets:\n{}".format(query, listing),
                        model=cfg.fast_model, effort="low", max_output_tokens=16000, timeout=420, label="AI ranking")
        data = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
    except (ProviderError, ValueError) as err:
        ui.warn("ranking skipped ({}); showing repository order".format(err))
        return [(d, "") for d in results[:limit]]
    named = data.get("genes") if isinstance(data.get("genes"), list) else []
    genes = [str(g).strip().upper() for g in named if re.match(r"^[A-Za-z0-9.-]{1,20}$", str(g))]
    rank.last_genes = genes[:2]
    by_acc = {d.accession: d for d in results}
    out = []
    for r in data.get("ranked") if isinstance(data.get("ranked"), list) else []:
        if isinstance(r, str):  # a bare accession where {"accession": ..., "why": ...} was asked for
            r = {"accession": r}
        if isinstance(r, dict) and isinstance(r.get("accession"), str) and r["accession"] in by_acc:
            out.append((by_acc[r["accession"]], str(r.get("why") or "")))
    seen, ordered = set(), []
    for d, why in out:
        if d.accession not in seen:
            seen.add(d.accession)
            ordered.append((d, why))
    # The studies the AI itself put in order. The ones below were shown to it and left out; their place in the
    # list returned here is the repository's, not a rank (a batch remembers only the AI's order).
    rank.ai_listed = [d.accession for d, _why in ordered]
    for d in results:  # keep the rest as fallbacks so a batch can look further down
        if d.accession not in seen:
            seen.add(d.accession)
            ordered.append((d, ""))
    return ordered[:limit]


def cmd_search(cfg, args):
    query = " ".join(getattr(args, "query", None) or []) or ui.ask("Search GEO + ArrayExpress for: ")
    if not query:
        return 1
    ui.step("Searching GEO and ArrayExpress: " + query)
    results = []
    for label, fn in (("GEO", repos.search_geo), ("ArrayExpress", repos.search_arrayexpress)):
        try:
            found = fn(query)
            ui.info("{}: {} expression datasets".format(label, len(found)))
            results += found
        except net.NetError as err:
            ui.warn("{} search failed: {}".format(label, err))
    if not results:
        ui.fail("Nothing found.")
        return 1
    min_samples = int(getattr(args, "min_samples", 0) or 0)
    kept = [d for d in results if not (d.n_samples and int(d.n_samples) < min_samples)]
    if len(kept) < len(results):
        ui.info("left out {} studies with fewer than {} samples (--min-samples)".format(
            len(results) - len(kept), min_samples))
    kept, _big = _cap_samples(kept, getattr(args, "max_samples", 0))
    results = kept or results
    llm = get_llm(cfg)
    ranked = rank(cfg, llm, query, results)
    ui.step("Results")
    for i, (d, why) in enumerate(ranked, 1):
        print("  {:>2}. {}".format(i, d.line()))
        if why:
            print("        why: " + why)
        if d.note:
            print("        note: " + d.note)
    choice = ui.ask("Pick a number, type an accession, or q to quit: ")
    if choice.lower() in ("", "q"):
        return 0
    if choice.isdigit() and 1 <= int(choice) <= len(ranked):
        ds = ranked[int(choice) - 1][0]
        return run_build(cfg, args, ds.accession, llm, ds)["code"]
    return run_build(cfg, args, choice, llm)["code"]


PICK_INSTRUCTIONS = """You choose which of a study's files to download to build ONE Hegemon expression dataset
(genes x samples, plus per-sample metadata), the way a lab member picks files by hand. Downloads are slow, so pick
the smallest set that holds expression values for all of the study's samples:
- RNA-seq: prefer raw counts (the lab notebook normalizes counts itself); take a normalized matrix only when there
  are no raw counts.
- Never pick two files with the same values in different units (raw and normalized counts, TPM and FPKM).
- When the samples are split across files (tissues, cohorts, batches, discovery and replication), pick every part.
- Skip files that are not per-sample expression values (gene lists, differential-expression results, QC tables)
  unless nothing else exists.
Reply with JSON only: {"download": ["exact names as listed"], "why": "one short sentence"}"""


def make_picker(cfg, llm):
    """The fast model picks between alternative data files before they are downloaded."""
    if llm is None:
        return None

    def picker(ds, files, what):
        listing = "\n\n".join("{}  ({})\n{}".format(
            name, "{:.1f} MB".format(size / 1e6) if size else "size unknown",
            "\n".join("    " + line for line in (head or "(first lines not available)").splitlines()))
            for name, size, head in files)
        text = "Study {}: {}\n{}\n\nThe {} to choose from:\n\n{}".format(
            ds.accession, ds.title, (ds.summary or "")[:1200], what, listing)
        reply = llm.ask(PICK_INSTRUCTIONS, text, model=cfg.fast_model, effort="low", max_output_tokens=4000,
                        timeout=180, label="AI choosing files")
        data = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
        names = {name for name, _size, _head in files}
        chosen = [str(n) for n in data.get("download", []) if str(n) in names]
        return chosen, str(data.get("why", ""))[:200]
    return picker


def _cap_samples(found, max_samples):
    """--max-samples: leave out the biggest studies (they take longest to process). 0 means no limit."""
    max_samples = int(max_samples or 0)
    if not max_samples:
        return found, []
    big = [d for d in found if d.n_samples and int(d.n_samples) > max_samples]
    if big:
        ui.info("left out {} studies with more than {} samples (--max-samples)".format(len(big), max_samples))
    return [d for d in found if d not in big], big


def run_build(cfg, args, accession, llm=None, ds=None, publish_key=None, link_label=None, genes=None):
    accession = accession.strip().upper()
    urls = list(getattr(args, "url", None) or [])
    min_samples = int(getattr(args, "min_samples", None) or 0)
    repo = repos.detect_repository(accession, allow_other=bool(urls or getattr(args, "source", None)))
    script = getattr(args, "script", None)
    if llm is None:
        llm = get_llm(cfg, required=not script)
    if ds is None:
        ds = repos.Dataset(accession, repo, accession) if (repo == "Other" or os.environ.get("HEGEMON_OFFLINE") == "1") \
            else repos.describe(accession)
    work = cfg.runs_root / "builds" / "{}-{}".format(accession, time.strftime("%Y%m%d-%H%M%S"))
    work.mkdir(parents=True, exist_ok=True)
    source = Path(args.source).resolve() if getattr(args, "source", None) else work / "source"
    receipt = {"agent_version": VERSION, "accession": accession, "repository": repo, "title": ds.title,
               "work": str(work), "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "model": cfg.model}
    clock = ui.Clock()
    ai_start = (llm.calls, llm.input_tokens, llm.output_tokens, llm.cached_tokens, llm.reasoning_tokens) \
        if llm else (0, 0, 0, 0, 0)
    ui.step("{}  {}".format(accession, ds.title))
    ui.info("build folder: " + str(work))

    notes, soft, gse, processed = [], None, None, []
    try:
        if urls:
            ui.step("Downloading {} URL(s)".format(len(urls)))
            notes, processed = repos.fetch_urls(ds, source, cfg, urls, getattr(args, "yes", False))
            if repo == "GEO":
                soft = next((p for p in source.rglob("*_family.soft*") if p.is_file()), None)
                if soft is not None:
                    gse = read_geo_soft(soft, sample_tables=2)
        elif getattr(args, "source", None):
            ui.info("using files already in " + str(source))
            if repo == "GEO":
                soft = next((p for p in (source / "{}_family.soft.gz".format(accession),
                                         source / "{}_family.soft".format(accession)) if p.is_file()), None)
                if soft is None:
                    return finish(work, receipt, "MISSING_SOFT", "{} needs {}_family.soft.gz".format(source, accession))
                gse = read_geo_soft(soft, sample_tables=2)
        else:
            ui.step("Downloading")
            max_samples = int(getattr(args, "max_samples", 0) or 0)
            picker = make_picker(cfg, llm)
            if repo == "GEO":
                soft, gse, notes, processed = repos.fetch_geo(ds, source, cfg, getattr(args, "yes", False),
                                                              min_samples=min_samples, max_samples=max_samples,
                                                              picker=picker)
            else:
                notes, processed = repos.fetch_arrayexpress(ds, source, cfg, getattr(args, "yes", False),
                                                            min_samples=min_samples, max_samples=max_samples,
                                                            picker=picker)
    except repos.SingleCell as err:
        return finish(work, receipt, "SINGLE_CELL", "{}; decided before downloading its data.".format(err))
    except repos.NotHosted as err:
        return finish(work, receipt, "NOT_HOSTED", str(err))
    except repos.FilesTooLarge as err:
        return finish(work, receipt, "FILES_TOO_LARGE", "{}. Raise HEGEMON_MAX_FILE_GB / HEGEMON_MAX_TOTAL_GB, "
                      "or pass --yes, to include them.".format(err))
    except repos.TooManySamples as err:
        return finish(work, receipt, "TOO_MANY_SAMPLES", "{} (decided from the sample table, before downloading its "
                      "data)".format(err))
    except repos.TooFewSamples as err:
        return finish(work, receipt, "TOO_FEW_SAMPLES", "{} ({} was decided from the sample table, so no data was "
                      "downloaded)".format(err, accession))
    except net.NetError as err:
        return finish(work, receipt, "DOWNLOAD_FAILED", str(err))
    for note in notes:
        ui.info(note)
    clock.lap("download")

    platform, prefix = None, re.sub(r"[^A-Za-z0-9._-]", "_", accession)
    if repo == "GEO" and gse is not None:
        counts = Counter((g.metadata.get("platform_id") or ["?"])[0] for g in gse.gsms.values())
        platform = (getattr(args, "platform", None) or "").upper() or None
        if platform is None and len(counts) == 1:
            platform = next(iter(counts))
        elif platform is None:
            ui.info("This series spans several platforms: " + ", ".join(
                "{} ({} samples)".format(p, n) for p, n in counts.most_common()))
            default = counts.most_common(1)[0][0]
            platform = ui.ask("Platform to build [{}]: ".format(default), default).upper()
        if platform not in counts:
            return finish(work, receipt, "BAD_PLATFORM", "{} is not in this series ({})".format(platform, dict(counts)))
        prefix = "{}-{}".format(accession, platform)  # the notebook's file_prefix
        max_samples = int(getattr(args, "max_samples", 0) or 0)
        if max_samples and counts[platform] > max_samples:
            return finish(work, receipt, "TOO_MANY_SAMPLES", "{} has {} samples on {}, more than --max-samples {}"
                          .format(accession, counts[platform], platform, max_samples))
        if ds.title == accession:
            ds.title = receipt["title"] = (gse.metadata.get("title") or [accession])[0]
        ds.kind = ds.kind or "; ".join(gse.metadata.get("type", []))
        ds.summary = ds.summary or (gse.metadata.get("summary") or [""])[0]
    run_dir = work / prefix  # folder named like the prefix, as jupyter_gse_processing expects
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt.update(prefix=prefix, platform=platform, run_dir=str(run_dir), source=str(source))

    if not getattr(args, "source", None) and not processed:
        size_note = next((n for n in notes if "NOT downloaded because of size" in n), "")
        offered = [e for n, e in ds.catalog.items() if not (source / n).is_file() and "raw" not in e.note.lower()]
        if ds.failures or offered:
            # the data did not arrive, or the study's files are not in a layout the fixed plan knows:
            # the AI sees everything the study offers and asks for the files it needs
            ui.info("no data file yet; the AI picks from the {} file(s) the study offers".format(len(ds.catalog)))
        elif size_note:
            return finish(work, receipt, "FILES_TOO_LARGE",
                          "{} has processed data, but it was not downloaded because of the size limits. {}"
                          .format(accession, size_note))
        else:
            return finish(work, receipt, "NO_PROCESSED_DATA",
                          "{} exposes no processed expression file; only raw data or nothing at all was found in "
                          "its public files.".format(accession))

    ui.step("Reading the files")
    try:
        _, series_samples, ev = evidence.build(ds, source, platform, soft, notes, gse=gse)
        if processed:
            ev += "\n\nPROCESSED FILES THE AGENT IDENTIFIED: " + ", ".join(str(x) for x in processed[:40])
    except Exception as err:
        return finish(work, receipt, "UNREADABLE_FILES", "Could not read the downloaded files: {}".format(err))
    (work / "evidence.txt").write_text(ev)
    ui.info("what the AI sees: {} ({:,} characters)".format(work / "evidence.txt", len(ev)))
    clock.lap("reading files")

    # A GEO microarray whose tables are all in the SOFT file is exactly the notebook's fixed
    # microarray path, so the built-in script runs first and the AI is only called if it fails.
    builtin = None
    if repo == "GEO" and gse is not None and platform:
        on_platform = [g for g in gse.gsms.values() if (g.metadata.get("platform_id") or [""])[0] == platform]
        kinds = " ".join(gse.metadata.get("type", [])).lower()

        def rows(g):
            try:
                return int((g.metadata.get("data_row_count") or ["0"])[0] or 0)
            except ValueError:
                return 0
        if on_platform and "sequencing" not in kinds and all(rows(g) > 0 for g in on_platform):
            builtin = convert.BUILTIN_GEO_ARRAY

    try:
        meta_source = None
        if repo == "GEO" and gse is not None:
            meta_source = {"kind": "GEO", "gse": gse}
        elif repo == "ArrayExpress":
            sdrf = next((p for p in Path(source).rglob("*.sdrf.txt") if p.is_file()), None)
            meta_source = {"kind": "ArrayExpress", "sdrf": sdrf} if sdrf else None
        fetched = []

        def fetcher(items):
            """The AI asked for files: download them and show it the evidence again."""
            report, got = repos.fetch_requested(ds, source, cfg, items, getattr(args, "yes", False))
            fetched.extend(got)
            processed.extend(got)
            _, _, new_ev = evidence.build(ds, source, platform, soft, notes, gse=gse)
            if processed:
                new_ev += "\n\nPROCESSED FILES THE AGENT IDENTIFIED: " + ", ".join(str(x) for x in processed[:40])
            (work / "evidence.txt").write_text(new_ev)
            receipt["fetched_by_ai"] = list(fetched)
            return "\n".join(report) or "nothing was downloaded", new_ev
        _, result, output, attempts = convert.convert(ds, prefix, work, run_dir, source, ev, series_samples,
                                                      llm, cfg, script, meta_source=meta_source, builtin=builtin,
                                                      fetcher=fetcher)
    except convert.Unconvertible as err:
        return finish(work, receipt, err.reason, err.detail)
    except convert.GaveUp as err:
        return finish(work, receipt, "AGENT_GAVE_UP",
                      "No working conversion after {} attempts. That is a limit of this agent, not proof the "
                      "dataset can't be converted. Last problem:\n{}".format(cfg.attempts, str(err)[-1500:]))
    except ProviderError as err:
        return finish(work, receipt, "AI_PROVIDER_" + err.kind.upper(), str(err) + " (an AI-provider problem, not the dataset)")
    clock.lap("conversion (built-in, no AI)" if attempts == 0 else "conversion (AI, {} attempt{})".format(
        attempts, "" if attempts == 1 else "s"))
    s = result.stats
    if min_samples and s["samples"] < min_samples:
        return finish(work, receipt, "TOO_FEW_SAMPLES", "{} samples, fewer than --min-samples {}".format(
            s["samples"], min_samples))
    ui.ok("files pass the checks: {} samples x {} probes, values {} to {}, {} metadata columns".format(
        s["samples"], s["probes"], s["min"], s["max"], s["metadata_columns"]))
    for col, text in s.get("groups", [])[:4]:
        ui.ok("groups to compare by '{}': {}".format(col, text))
    for w in result.warnings:
        ui.warn(w)
    for line in [l for l in output.strip().splitlines() if l.strip()][-8:]:
        ui.info("script: " + line[:200])

    ui.step("The notebooks' last two cells: idx + jupyter_gse_processing")
    n = native.make_idx(run_dir, prefix)
    problem = native.check_idx(run_dir, prefix)
    if problem:
        return finish(work, receipt, "IDX_FAILED", problem)
    ui.ok("idx: {} rows; every pointer lands on its expr row".format(n))
    if not cfg.lab_script.is_file():
        return finish(work, receipt, "LAB_SCRIPT_MISSING", "Not found: {} (set HEGEMON_BOOLEANLAB_SCRIPT)".format(cfg.lab_script))
    log = work / "logs" / "jupyter_gse_processing.log"
    ui.info("bash {} {}   (in {})".format(cfg.lab_script, prefix, run_dir))
    rc = native.run_lab_script(cfg, run_dir, prefix, log)
    if rc != 0:
        return finish(work, receipt, "LAB_SCRIPT_FAILED",
                      "jupyter_gse_processing exited with {}. End of {}:\n{}".format(rc, log, native.tail(log)))
    support = native.check_support(run_dir, prefix, s["samples"])
    for kind, text in support.stats.items():
        ui.info("{}: {}".format(kind, text))
    for w in support.warnings:
        ui.warn(w)
    if not support.ok:
        return finish(work, receipt, "SUPPORT_FILES_BAD", support.text())
    ui.ok("thr/info/vinfo/bv read back and match this expr file")
    clock.lap("idx + lab script")
    native.open_permissions(run_dir, cfg.runs_root)

    lab_hash = hashlib.sha256(cfg.lab_script.read_bytes()).hexdigest()
    receipt.update(status="built", samples=s["samples"], probes=s["probes"], value_range=[s["min"], s["max"]],
                   groups=s.get("groups", []),
                   metadata_columns=s["metadata_columns"], warnings=result.warnings + support.warnings,
                   attempts=attempts, script=str(work / "convert.py"), lab_script=str(cfg.lab_script),
                   lab_script_sha256=lab_hash, ai_calls=(llm.calls - ai_start[0]) if llm else 0,
                   ai_tokens=[llm.input_tokens - ai_start[1], llm.output_tokens - ai_start[2]] if llm else [0, 0],
                   ai_cached_tokens=(llm.cached_tokens - ai_start[3]) if llm else 0,
                   ai_reasoning_tokens=(llm.reasoning_tokens - ai_start[4]) if llm else 0,
                   sha256=native.file_hashes(run_dir, prefix), built_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    (work / "build.json").write_text(json.dumps(receipt, indent=2))
    ui.step("Built " + prefix)
    ui.info("files: " + str(run_dir))
    calls, (tin, tout) = receipt["ai_calls"], receipt["ai_tokens"]
    ui.info("AI use: {} call(s), {:,} input / {:,} output tokens{}".format(
        calls, tin, tout, " (built-in notebook conversion)" if attempts == 0 else ""))
    result = {"status": "built", "receipt": receipt, "work": str(work), "code": 0}

    def timing():
        receipt["timing"] = {name: round(sec, 1) for name, sec in clock.laps}
        receipt["timing"]["total"] = round(clock.total(), 1)
        (work / "build.json").write_text(json.dumps(receipt, indent=2))
        ui.info("time: " + clock.text())

    if getattr(args, "no_publish", False):
        timing()
        ui.info("publish later with: hegemon publish " + str(work))
        return result
    if publish_key or ui.yes_no("Publish {} to {} now?".format(prefix, cfg.site_url), False,
                                True if getattr(args, "yes", False) else None):
        genes = genes or [g.strip().upper() for g in (getattr(args, "genes", None) or "").split(",") if g.strip()]
        code = do_publish(cfg, receipt, work, key=publish_key, link_label=link_label, genes=genes[:2])
        result["code"] = code
        result["status"] = "published" if code == 0 else "publish_failed"
        result["detail"] = receipt.get("publish_error", "")
        result["url"] = (receipt.get("published") or {}).get("url")
        clock.lap("publish")
        timing()
        return result
    timing()
    ui.info("publish later with: hegemon publish " + str(work))
    return result


def do_publish(cfg, receipt, work, key=None, link_label=None, genes=None):
    ui.step("Publishing to " + cfg.site_url + (" under key '{}'".format(key) if key else ""))
    if native.file_hashes(receipt["run_dir"], receipt["prefix"]) != receipt["sha256"]:
        ui.fail("The files changed after the build was checked. Rebuild instead of publishing them.")
        return 1
    try:
        section, key, url = publish.register(cfg, receipt, key=key, link_label=link_label, genes=genes)
    except publish.PublishError as err:
        ui.fail(str(err))
        receipt["publish_error"] = str(err)
        (Path(work) / "build.json").write_text(json.dumps(receipt, indent=2))
        return 1
    receipt["published"] = {"section": section, "key": key, "url": url, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (Path(work) / "build.json").write_text(json.dumps(receipt, indent=2))
    ui.ok("live: " + url)
    return 0


def cmd_build(cfg, args):
    return run_build(cfg, args, args.accession)["code"]


# --------------------------------------------------------------------------- batch

def batch_state_path(cfg, key):
    root = cfg.runs_root / "batches"
    root.mkdir(parents=True, exist_ok=True)
    return root / "{}.json".format(re.sub(r"[^A-Za-z0-9._-]", "_", key))


def published_under_key(cfg, key):
    """Accessions already carrying this key in explore.conf: the source of truth."""
    conf = cfg.site_root / "explore.conf"
    if not conf.is_file():
        return {}
    out = {}
    for section, values in publish._sections(conf.read_text(errors="replace")).items():
        if key in [k.strip() for k in values.get("key", "").split(":")]:
            out[values.get("source", "").strip().upper()] = section
    return out


def search_id(query):
    """Two searches are the same search when their words are: capitals and spacing do not matter.

    Everything else does. "ER+ breast cancer" and "ER- breast cancer" are different questions.
    """
    return " ".join(str(query or "").lower().split())


def slot_in(saved, fresh, listed=None):
    """The remembered ranking, with the studies it has not seen put in where today's ranking has them.

    saved:  accessions best first, as this search ranked them before.
    fresh:  today's order of the studies the search returned today: the ones the AI ranked, then the rest.
    listed: the ones the AI ranked today (default: all of `fresh`).
    Nothing in `saved` moves, so asking for the same thing again never reshuffles a page. A study that is not in
    `saved` goes in at the rank the AI gave it today: 4th in `fresh` is 4th among today's studies in the result.
    A new study the AI was shown and left out has no rank, so it goes last.
    """
    order, known, today = list(saved), set(saved), set(fresh)
    ranked_today = today if listed is None else set(listed)
    here = [a for a in order if a in today]  # ranked before and returned again, in their remembered order
    last = []
    for pos, acc in enumerate(fresh):
        if acc in known:
            continue
        known.add(acc)
        if acc not in ranked_today:
            last.append(acc)
        elif pos < len(here):
            order.insert(order.index(here[pos]), acc)
            here.insert(pos, acc)
        else:
            order.append(acc)
            here.append(acc)
    return order + last


# Outcomes that are a limit's doing, not the dataset's, and the limit that decides each. The limit in force is kept
# with the outcome, so the dataset is tried again once that limit lets more through.
LIMIT_OUTCOMES = {"TOO_MANY_SAMPLES": "max_samples", "TOO_FEW_SAMPLES": "min_samples", "FILES_TOO_LARGE": "size"}


def batch_limits(cfg, args):
    """The limits that can turn a dataset away on their own, as this run has them."""
    return {"min_samples": int(getattr(args, "min_samples", 0) or 0),
            "max_samples": int(getattr(args, "max_samples", 0) or 0),
            "size": [cfg.max_file_gb, cfg.max_total_gb, bool(getattr(args, "yes", False))]}


def _loosened(name, then, now):
    """True when a limit lets more through now than when it turned a dataset away."""
    try:
        if name == "max_samples":
            return bool(then) and (not now or now > then)
        if name == "min_samples":
            return now < then
        return bool(now[2] and not then[2]) or now[0] > then[0] or now[1] > then[1]
    except (TypeError, IndexError, KeyError):
        return False


def failed_before(tried, acc, limits=None):
    """The outcome that keeps a dataset this page already tried from being tried again, or None.

    Anything that was tried and did not publish counts, whatever the reason: asking for the same thing again
    must not spend minutes on the same dataset failing the same way (--retry-failed does that on request).
    Two exceptions, because in both the dataset has not had its try: an OpenAI problem, and a limit that turned
    the dataset away and has been loosened since (--max-samples raised, for one).
    """
    entry = tried.get(acc)
    entry = entry if isinstance(entry, dict) else {}
    status = str(entry.get("status") or "")
    if not status or status == "published" or status.startswith("AI_PROVIDER_"):
        return None
    name = LIMIT_OUTCOMES.get(status)
    if name and limits and "limit" in entry and _loosened(name, entry["limit"], limits[name]):
        return None
    return status


def save_batch_state(path, state):
    """The batch record, replaced in one step: a run that is cut off cannot leave half a file behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(str(tmp), str(path))


def batch_ranking(cfg, llm, state, query, found, tried, limit, rerank=False):
    """Accessions best first for this search, and the AI's reason for each: (order, why).

    The ranking a search gets is remembered in the batch record, so asking for the same thing again gives the
    same order and the best 10 stay the best 10. A study the search has not returned before is ranked by the AI
    and goes in at the rank it gets (slot_in); nothing ranked earlier moves. When the search returns no such
    study the AI is not asked at all. --rerank forgets the remembered ranking and takes the AI's order today.
    """
    sid = search_id(query)
    rankings = state.get("rankings") if isinstance(state.get("rankings"), dict) else {}
    memory = rankings.get(sid) if isinstance(rankings.get(sid), dict) else {}
    saved = [str(a).upper() for a in memory.get("order") or []]
    reasons = memory.get("why") if isinstance(memory.get("why"), dict) else {}
    why = {str(a).upper(): str(r) for a, r in reasons.items()} if saved else {}
    since = memory.get("ranked_at") or "the last run"
    history = False
    if not saved:
        # No ranking is remembered for this search, but it may have tried datasets (a page made before rankings
        # were remembered, or a run when the AI could not rank). The order it tried them in is the top of the
        # ranking it had then, so they keep it. Datasets another search put on this page are not part of it.
        saved = [acc for acc, entry in tried.items() if isinstance(entry, dict) and entry.get("search") == sid]
        history = bool(saved)
    known = set(saved)
    pool = [d.accession.upper() for d in found[:limit]]  # the studies the AI is shown
    new = [a for a in pool if a not in known]
    rank.last_genes, rank.ai_listed = [], []
    if saved and not new and not history and not rerank:
        ui.info("{} already ranked ({}), so that ranking stands".format(
            "every study this search returned was" if len(found) <= limit else
            "the first {} studies this search returned were all".format(limit), since))
        return saved, why
    ranked = rank(cfg, llm, query, found, limit=limit)
    # today's order: what the AI ranked of the studies it was shown, then the rest as the search gave them
    listed = [a for a in dict.fromkeys(str(x).upper() for x in rank.ai_listed) if a in set(pool)]
    fresh = listed + [a for a in pool if a not in set(listed)]
    if not listed:
        # The AI ranked nothing (it could not be reached, or its reply was no ranking). Repository order is not a
        # ranking: nothing is remembered from it, and the studies that are new stay new for the next run. They
        # go last here, because their place in a list nobody ranked means nothing.
        if rerank and saved:
            ui.warn("the AI did not rank, so --rerank changed nothing: the ranking this search already has is used")
        return saved + [a for a in fresh if a not in known], why
    if rerank:
        order, why = fresh, {}
    elif history:
        order = saved + [a for a in fresh if a not in known]
        ui.info("the datasets this search already tried keep the order they were tried in; the other studies "
                "follow in today's ranking")
    else:
        order = slot_in(saved, fresh, listed)
        if saved:
            got, left = [a for a in new if a in set(listed)], [a for a in new if a not in set(listed)]

            def names(items, show=str):
                return ", ".join(show(a) for a in items[:6]) + (
                    " and {} more".format(len(items) - 6) if len(items) > 6 else "")
            text = "{} stud{} new to this search since {}.".format(
                len(new), "y is" if len(new) == 1 else "ies are", since)
            if got:
                text += " The AI ranked {} among the {} it was shown: {}.".format(
                    "it" if len(new) == 1 else len(got), len(pool),
                    names(got, lambda a: "{} (#{})".format(a, fresh.index(a) + 1)))
            if left:
                text += " {} left {} out of its ranking, so {} last: {}.".format(
                    "It" if got else "The AI", "it" if len(new) == 1 else len(left),
                    "it goes" if len(left) == 1 else "they go", names(left))
            ui.info(text)
    for d, reason in ranked:
        if reason:
            why.setdefault(d.accession.upper(), str(reason)[:200])  # a study keeps the reason it was first ranked for
    rankings[sid] = {"query": query, "order": order, "why": why, "ranked_at": time.strftime("%Y-%m-%d")}
    state["rankings"] = rankings
    return order, why


def cmd_batch(cfg, args):
    key = args.key.strip()
    if not re.match(r"^[A-Za-z0-9._+-]+$", key):
        raise SystemExit("A key may only contain letters, digits, '.', '_', '+' or '-' (Hegemon splits keys on ':').")
    for bad in ("source", "script"):
        if getattr(args, bad, None):
            raise SystemExit("--{} applies to one dataset, so it cannot be used with batch.".format(bad))
    args.no_publish = False  # a batch publishes as it goes, so partial progress is usable
    state_path = batch_state_path(cfg, key)
    try:
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if not isinstance(state, dict):
            raise ValueError("it does not hold a record")
    except ValueError as err:
        raise SystemExit("The batch record {} cannot be read ({}). Move it aside and run again: what is on the page "
                         "is read from explore.conf, not from this file.".format(state_path, err))
    query = " ".join(args.query or []).strip() or state.get("query", "")
    if not query:
        raise SystemExit("Give a search query the first time you use a key: hegemon batch \"...\" --key " + key)
    want = int(args.count)
    if want < 1:
        raise SystemExit("--count must be 1 or more.")
    label = args.label or query
    # Default: the page should hold the best `want` datasets for this search, so what is already there counts and
    # only what is new gets built. --more: add the next `want` down the ranking, whatever is already there.
    more, retry = bool(getattr(args, "more", False)), bool(getattr(args, "retry_failed", False))
    sid, limits = search_id(query), batch_limits(cfg, args)
    # the command that repeats this request, for the hints printed at the end
    if getattr(args, "cmd", None) == "agent":
        again = "hegemon agent" + ("" if key == key_from_query(query) else " --key " + key)
    else:
        again = "hegemon batch --key " + key
    count = "" if want == 10 else " --count {}".format(want)

    ui.step("Batch '{}': {} dataset(s) for: {}".format(key, want, query))
    already = published_under_key(cfg, key)
    if already:
        ui.info("{} dataset(s) already published under '{}': {}".format(len(already), key, ", ".join(sorted(already))))
        ui.info("This run adds {} more and skips those.".format(want) if more else
                "Those are skipped. This run adds only what is new in the best {} for this search.".format(want))
    tried = {a.upper(): v for a, v in (state.get("tried") or {}).items()}
    for entry in tried.values():
        if isinstance(entry, dict):  # a record from before 1.1 does not say which search tried a dataset:
            entry.setdefault("search", search_id(state.get("query")))  # it was the one search the record names

    llm = get_llm(cfg)
    ui.step("Searching GEO and ArrayExpress")
    found = []
    for name, fn in (("GEO", repos.search_geo), ("ArrayExpress", repos.search_arrayexpress)):
        try:
            hits = fn(query, limit=100)
            ui.info("{}: {} expression datasets".format(name, len(hits)))
            found += hits
        except net.NetError as err:
            ui.warn("{} search failed: {}".format(name, err))
    if not found:
        ui.fail("Nothing found for: " + query)
        return 1
    single = [d for d in found if repos.looks_single_cell(d.title)]
    found = [d for d in found if not repos.looks_single_cell(d.title)]
    if single:
        ui.info("left out {} single-cell studies (no single-cell blueprint yet)".format(len(single)))
    min_samples = int(args.min_samples or 0)
    small = [d for d in found if d.n_samples and int(d.n_samples) < min_samples]
    found = [d for d in found if not (d.n_samples and int(d.n_samples) < min_samples)]
    if small:
        ui.info("left out {} studies with fewer than {} samples (--min-samples)".format(len(small), min_samples))
    found, _big = _cap_samples(found, getattr(args, "max_samples", 0))
    if not found:
        ui.fail("Nothing left with at least {} samples. Lower --min-samples or change the query.".format(min_samples))
        return 1
    limit = max(60, want * 6)
    order, why = batch_ranking(cfg, llm, state, query, found, tried, limit, rerank=bool(getattr(args, "rerank", False)))
    pool = {d.accession.upper(): d for d in found[:limit]}  # the studies the ranking was about
    genes = [g.strip().upper() for g in (args.genes or "").split(",") if g.strip()][:2] or \
        rank.last_genes or state.get("genes") or []
    if genes:
        state["genes"] = genes
        ui.info("page opens on {} vs {} when both are in every dataset (--genes A,B to change)".format(*genes)
                if len(genes) == 2 else "")

    def standing(acc):
        """Where a ranked study stands for this page: published, gone, failed or new."""
        if acc in already:
            return "published"   # explore.conf says so: the source of truth
        if acc not in pool:
            return "gone"        # ranked on an earlier run; the search did not return it today
        if failed_before(tried, acc, limits) and not retry:
            return "failed"
        return "new"

    def outcome(result, dataset):
        """What the batch record keeps about one try."""
        status = result.get("status")
        entry = {"status": status, "detail": str(result.get("detail", ""))[:400], "work": result.get("work", ""),
                 "at": time.strftime("%Y-%m-%d %H:%M:%S"), "title": dataset.title, "search": sid}
        if status in LIMIT_OUTCOMES:
            entry["limit"] = limits[LIMIT_OUTCOMES[status]]
        return entry

    # The plan, before anything is built: walk down the ranking until `want` places are taken.
    on_page, to_add, skipped = [], [], []
    for acc in order:
        if len(on_page) + len(to_add) >= want:
            break
        what = standing(acc)
        if what == "published" and not more:
            on_page.append(acc)
        elif what == "failed":
            skipped.append(acc)
        elif what == "new":
            to_add.append(acc)
    if more:
        ui.info("the next {} down the ranking: {}{}".format(
            len(to_add), ", ".join(to_add) or "nothing left",
            " (all the ranking has left)" if 0 < len(to_add) < want else ""))
    elif already or skipped:
        ui.info("the best {} for this search: {} already on your page, {} to add{}".format(
            want, len(on_page), len(to_add), ": " + ", ".join(to_add) if to_add else ""))
    else:
        ui.info("{} ranked candidates, {} to try".format(
            len(pool), sum(1 for acc in order if standing(acc) == "new")))
    state.update(key=key, query=query, label=label, tried=tried, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_batch_state(state_path, state)  # the ranking is kept even when nothing gets built

    # `filled` counts places taken in the best `want`; `goal` is how many of them this run has to add.
    published, failed, url, filled, attempted, goal = [], [], None, 0, set(), len(to_add)
    for acc in (order if to_add else []):
        if filled >= want:
            break
        what = standing(acc)
        if what == "published":
            filled += 0 if more else 1
            continue
        if what == "gone":
            continue
        if what == "failed":
            ui.info("skipping {} ({} last time; --retry-failed to try again)".format(
                acc, failed_before(tried, acc, limits)))
            continue
        dataset = pool[acc]
        attempted.add(acc)
        size = "{} samples, ".format(dataset.n_samples) if dataset.n_samples else ""
        ui.step("[{}/{}] {}  {}{}".format(len(published) + 1, goal, acc, size, dataset.title[:80]))
        if why.get(acc):
            ui.info("ranked because: " + str(why[acc]))
        try:
            result = run_build(cfg, args, acc, llm, dataset, publish_key=key, link_label=label, genes=genes)
        except ProviderError as err:
            ui.fail("Stopping the batch: " + str(err))
            ui.info("This is an AI-provider problem, not a dataset problem. Nothing published so far is affected.")
            break
        except KeyboardInterrupt:
            raise
        except Exception as err:  # never let one dataset end the batch
            result = {"status": "AGENT_ERROR", "detail": repr(err), "work": "", "code": 2}
            ui.fail("unexpected agent error on {}: {!r}".format(acc, err))
        status = result.get("status")
        tried[acc] = outcome(result, dataset)
        if str(status).startswith("AI_PROVIDER_"):
            ui.fail("Stopping the batch: " + str(result.get("detail", status)))
            ui.info("This is an AI-provider problem, not a dataset problem. Anything already published is unaffected.")
        elif status == "published":
            published.append((acc, result["receipt"]))
            filled += 1
            url = result.get("url") or url
            ui.ok("{} of {} published under '{}'".format(len(published), goal, key))
        else:
            failed.append((acc, status, str(result.get("detail", ""))[:200]))
        state.update(key=key, query=query, label=label, tried=tried,
                     updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_batch_state(state_path, state)
        if str(status).startswith("AI_PROVIDER_"):
            break

    total = published_under_key(cfg, key)
    link = url or "{}/explore.php?key={}".format(cfg.site_url, key)
    further = [acc for acc in order if acc not in attempted and standing(acc) == "new"]  # not looked at yet
    # only the first `limit` studies a search returns are ranked; say so where "nothing is left" would mislead
    beyond = ("The ranking covers the first {} of the {} studies the search returned; a larger --count ranks "
              "more of them.".format(limit, len(found)) if len(found) > limit else "")
    if to_add:
        ui.step("Batch '{}' finished".format(key))
        for acc, receipt in published:
            groups = "; ".join("{}: {}".format(c, g) for c, g in (receipt.get("groups") or [])[:2])
            took = ui.duration((receipt.get("timing") or {}).get("total", 0))
            ui.ok("{}  {} samples x {} probes | {} | {}".format(acc, receipt["samples"], receipt["probes"], took, groups))
        if failed:
            ui.info("Not published this run:")
            for acc, status, detail in failed:
                ui.info("  {}: {} - {}".format(acc, status, detail))
        if total:
            ui.step("{} dataset(s) are now on one page under key '{}'".format(len(total), key))
            ui.info(link)
            if not more:
                ui.info("Ask for the same thing again and only what is new in the best {} is added.".format(want))
    else:
        # Nothing was built: say so plainly, and say why.
        ui.step("Nothing new for '{}'".format(key))
        if more:
            ui.info("There is nothing further down the ranking for \"{}\": every study in it is on your page or "
                    "did not publish before.".format(query))
        elif len(on_page) >= want:
            ui.ok("The best {} datasets for \"{}\" are already on your page. Nothing was added.".format(want, query))
        elif on_page:
            ui.ok("All {} datasets this ranking can give you are already on your page. Nothing was added.".format(
                len(on_page)))
        if beyond and (more or len(on_page) < want):
            ui.info(beyond)
        if skipped:
            ui.info("{} that did not publish before {} left alone: {}{}".format(
                len(skipped), "was" if len(skipped) == 1 else "were",
                ", ".join("{} ({})".format(acc, failed_before(tried, acc, limits)) for acc in skipped[:8]),
                " and {} more".format(len(skipped) - 8) if len(skipped) > 8 else ""))
            ui.info("To try {} again: {}{}{} --retry-failed".format(
                "it" if len(skipped) == 1 else "them", again, " --more" if more else "", count))
        if total:
            ui.info("Your page ({} dataset{}): {}".format(len(total), "" if len(total) == 1 else "s", link))
    if further and total and more:
        ui.info("For the {} after these: {} --more{}".format(want, again, count))
    elif further and total:
        ui.info("For the next {} down the ranking: {} --more{}".format(want, again, count))
    if not total:
        ui.fail("Nothing was published under '{}'.".format(key))
    ui.info("Batch record: " + str(state_path))
    if more:
        return 0 if published else 1
    # the page holds the best it can for this search, whether this run added to it or found nothing new
    return 0 if published or filled >= want or (not to_add and on_page) else 1


_KEY_SKIP = {"a", "an", "and", "the", "of", "in", "on", "for", "to", "with", "from", "by", "or", "at", "vs",
             "versus", "dataset", "datasets", "data", "study", "studies"}


def key_from_query(query, limit=24):
    """The page key for a search typed into `hegemon agent`: its words run together, e.g.
    'psoriasis lesional skin' -> psoriasislesionalskin. The same words always give the same page."""
    words = re.findall(r"[a-z0-9]+", query.lower())
    words = [w for w in words if w not in _KEY_SKIP] or words
    key = ""
    for word in words:
        if key and len(key) + len(word) > limit:
            break
        key += word
    return key[:limit] or "datasets" + time.strftime("%m%d%H%M")


def cmd_agent(cfg, args):
    """`hegemon agent`: ask what to look for, then run the same 10-dataset batch as `hegemon batch`.

    This only changes how a batch is started. Everything after the question is cmd_batch.
    """
    if not ready_mode(cfg):
        return 1
    query = " ".join(args.query or []).strip()
    if not query:
        print()
        query = ui.ask("What datasets do you want to look for? ")
        if not query:
            ui.info("Type a topic, for example: psoriasis lesional skin")
            query = ui.ask("What datasets do you want to look for? ")
    query = query.strip().strip("\"'").strip()
    if not query:
        ui.fail("Nothing was typed, so nothing was started.")
        return 1
    args.query = [query]
    if not args.key:
        args.key = key_from_query(query)
    ui.info("Your page: {}/explore.php?key={}".format(cfg.site_url, args.key))
    return cmd_batch(cfg, args)


def choose_mode(cfg):
    """Ask local / mixed / cloud and save the answer. Returns False if nothing usable was chosen."""
    print()
    ui.info("Where should the AI run?")
    for i, m in enumerate(MODES, 1):
        ui.info("  {}. {}".format(i, MODE_TEXT[m]))
    current = cfg.mode or "cloud"
    answer = ui.ask("Type 1, 2 or 3 [{}]: ".format(MODES.index(current) + 1)).strip().lower()
    pick = {"1": "cloud", "2": "local", "3": "mixed"}.get(answer, answer if answer in MODES else current)
    settings = load_settings()
    cfg.mode = pick
    cfg.apply_mode()
    if pick != "cloud" and not setup_local(cfg, settings):
        return False
    settings["mode"] = cfg.mode
    save_settings(settings)
    ui.ok("saved: {} (change it any time with: hegemon mode)".format(MODE_TEXT[cfg.mode]))
    return True


def setup_local(cfg, settings):
    """Make sure the local model answers; ask for its address and name until it does, or switch to cloud."""
    llm = LLM(None, cfg)
    while True:
        try:
            ui.info("testing {} at {} ...".format(cfg.local_model, cfg.local_url))
            llm.ping(cfg.local_model)
            ui.ok("the local model answered")
            settings.update(local_url=cfg.local_url, local_model=cfg.local_model)
            return True
        except ProviderError as err:
            ui.fail(str(err))
        if not sys.stdin.isatty():
            return False
        ui.info("The server reaches your laptop's Ollama only through the ssh tunnel. Log in with:")
        ui.info("  ssh -R 11434:127.0.0.1:11434 {}@hegemon.ucsd.edu".format(cfg.user))
        ui.info("and keep Ollama running on the laptop (check there with: curl http://127.0.0.1:11434/v1/models).")
        choice = ui.ask("Enter to try again, a new address or model name to change it, or 'cloud' to use OpenAI: ")
        choice = choice.strip()
        if choice.lower() == "cloud":
            cfg.mode = "cloud"
            cfg.apply_mode()
            return True
        if choice.startswith("http"):
            cfg.local_url = choice.rstrip("/")
        elif choice:
            cfg.local_model = choice
        cfg.apply_mode()


def ready_mode(cfg):
    """Before a run: ask the mode the first time (then remember it); for local/mixed, check the model answers."""
    if not sys.stdin.isatty():
        return True                     # scripts and tests: the saved mode, or cloud as before
    if cfg.mode is None:
        return choose_mode(cfg)
    ui.info("AI: {} (change with: hegemon mode)".format(MODE_TEXT[cfg.mode]))
    if cfg.mode == "cloud":
        return True
    settings = load_settings()
    if not setup_local(cfg, settings):
        return False
    if cfg.mode == "cloud":             # switched to cloud during setup
        settings["mode"] = "cloud"
    save_settings(settings)
    return True


def cmd_mode(cfg, args):
    return 0 if choose_mode(cfg) else 1


SHELL_LINE = ('[ -n "$BASH_VERSION$ZSH_VERSION" ] && [ -f "$HOME/.hegemon-agent-commands.sh" ] && '
              '. "$HOME/.hegemon-agent-commands.sh"  # hegemon-agent command')


def cmd_install(cfg, args):
    """Make `hegemon` a command in your shell, so `hegemon agent` works straight after ssh. Run once."""
    import shlex
    import subprocess
    launcher = Path(os.path.abspath(__file__)).parent / "hegemon"   # as you reach it, not through symlinks
    home = Path.home()
    try:
        call = '"$HOME/{}"'.format(launcher.relative_to(home))
    except ValueError:
        call = shlex.quote(str(launcher))
    snippet = home / ".hegemon-agent-commands.sh"
    ui.step("Making `hegemon` a command for " + cfg.user)
    try:
        snippet.write_text(
            "# Written by `hegemon install`. Gives your shell the commands `hegemon` and `hegemon-agent`.\n"
            "unalias hegemon 2>/dev/null\n"
            "hegemon() {{ bash {0} \"$@\"; }}\n"
            "alias hegemon-agent='hegemon agent'\n".format(call))
        shell = os.path.basename(os.environ.get("SHELL", "") or "bash")
        login = next((home / n for n in (".bash_profile", ".bash_login", ".profile") if (home / n).is_file()),
                     home / ".profile")
        files = [home / ".bashrc", login]
        if shell == "zsh" or (home / ".zshrc").is_file():
            files.append(home / ".zshrc")
        for rc in files:
            text = rc.read_text(errors="replace") if rc.is_file() else ""
            if "# hegemon-agent command" in text:
                ui.info("{} already has it".format(rc))
                continue
            with open(str(rc), "a") as fh:
                fh.write(("" if text.endswith("\n") or not text else "\n") + SHELL_LINE + "\n")
            ui.info("added one line to {}".format(rc))
    except OSError as err:
        ui.fail("Could not write the shell setup ({}).".format(err))
        ui.info("Start the agent with this instead: bash {} agent".format(launcher))
        return 1
    if shell not in ("bash", "zsh"):
        ui.warn("Your login shell is {}, which this setup does not cover. Start the agent with: bash {} agent"
                .format(shell, launcher))
        return 1
    # prove it in a brand-new login shell, the way the next ssh session starts
    try:
        probe = subprocess.run([shell, "-lic", "type hegemon"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, universal_newlines=True, timeout=30)
        works = "function" in probe.stdout
    except (OSError, subprocess.TimeoutExpired):
        works = False
    if works:
        ui.ok("a new login shell knows the command `hegemon`")
    else:
        ui.warn("could not confirm it in a new login shell; if `hegemon agent` says 'command not found' after you "
                "log in again, start it with: bash {} agent".format(launcher))
    ui.step("Done")
    ui.info("Log out, ssh in again, and type:  hegemon agent")
    ui.info("(to use it in this window right now:  source ~/.hegemon-agent-commands.sh)")
    return 0


def cmd_diagnose(cfg, args):
    ui.step("Checking the datasets already on " + cfg.site_url)
    try:
        rows = publish.diagnose(cfg, args.key, min_samples=args.min_samples)
    except publish.PublishError as err:
        ui.fail(str(err))
        return 1
    if not rows:
        ui.info("explore.conf has no datasets" + (" under key '{}'".format(args.key) if args.key else ""))
        return 0
    broken = []
    for row in rows:
        title = "[{}] {} {}".format(row["section"], row["source"], row["name"][:60])
        if row["problems"]:
            broken.append(row)
            ui.fail(title)
            for problem in row["problems"]:
                ui.info("  " + problem)
        else:
            groups = "; ".join("{}: {}".format(c, g) for c, g in (row.get("groups") or [])[:2])
            ui.ok("{}  {} samples | {}".format(title, row.get("samples", "?"), groups or "groups ok"))
        for note in row["notes"]:
            ui.info("  note: " + note)
        if row["keys"]:
            ui.info("  {}/explore.php?key={}".format(cfg.site_url, row["keys"][0]))
    ui.step("{} of {} dataset(s) are fine".format(len(rows) - len(broken), len(rows)))
    if broken:
        ui.info("Take the broken ones off the site with:")
        ui.info("  hegemon unpublish " + " ".join(r["section"] for r in broken))
    return 1 if broken else 0


def cmd_unpublish(cfg, args):
    if not args.sections and not args.key:
        raise SystemExit("Say what to remove: hegemon unpublish --key KEY, or hegemon unpublish SECTION [SECTION ...]")
    try:
        remove, strip, values = publish.plan_unpublish(cfg, args.sections, args.key)
    except publish.PublishError as err:
        ui.fail(str(err))
        return 1
    if not remove and not strip:
        ui.fail("Nothing in explore.conf matches" + (" key '{}'".format(args.key) if args.key else ""))
        return 1
    ui.step("Will take off {}".format(cfg.site_url))
    for name in remove:
        ui.info("remove [{}] {}".format(name, values[name].get("name", "")[:80]))
    for name in strip:
        ui.info("keep [{}] but drop key '{}' from it (it has other keys)".format(name, args.key))
    ui.info("The data files are not deleted; only the site entries.")
    if not ui.yes_no("Go ahead?", False, True if args.yes else None):
        ui.info("Nothing changed.")
        return 1
    try:
        remove, strip, dead, links, backup = publish.unpublish(cfg, args.sections, args.key)
    except publish.PublishError as err:
        ui.fail(str(err))
        return 1
    ui.ok("removed {} dataset(s){}".format(len(remove), ", edited {}".format(len(strip)) if strip else ""))
    if links:
        ui.ok("removed {} homepage link(s) for: {}".format(links, ", ".join(dead)))
    ui.info("previous explore.conf and index.html saved in " + str(backup))
    return 0


def cmd_relink(cfg, args):
    genes = [g.strip().upper() for g in (args.genes or "").split(",") if g.strip()][:2]
    ui.step("Making each key's link open on a plot" + (" (key '{}')".format(args.key) if args.key else ""))
    try:
        rows = publish.relink(cfg, args.key, genes)
    except publish.PublishError as err:
        ui.fail(str(err))
        return 1
    if not rows:
        ui.fail("No datasets" + (" under key '{}'".format(args.key) if args.key else "") + " in explore.conf")
        return 1
    bad = 0
    for row in rows:
        url = "{}/{}".format(cfg.site_url, publish.key_link(row["key"], row["pair"]))
        if row["problem"]:
            bad += 1
            ui.fail("{} ({} dataset(s)): {}".format(row["key"], row["datasets"], row["problem"]))
        else:
            ui.ok("{} ({} dataset(s)): opens on {} vs {} ({})".format(row["key"], row["datasets"], *row["pair"]))
            ui.info("  " + url)
    return 1 if bad else 0


def cmd_publish(cfg, args):
    work = Path(args.build).resolve()
    receipt = json.loads((work / "build.json").read_text())
    if receipt.get("status") != "built":
        ui.fail("That build did not finish ({}).".format(receipt.get("status")))
        return 1
    genes = [g.strip().upper() for g in (args.genes or "").split(",") if g.strip()][:2]
    return do_publish(cfg, receipt, work, key=args.key, genes=genes)


def cmd_compare(cfg, args):
    same, report = compare_mod.compare(args.agent, args.lab, tol=args.tol)
    print(report)
    return 0 if same else 1


def cmd_key(cfg, args):
    from getpass import getpass
    key = getpass("   OpenAI API key (hidden): ").strip()
    if not key:
        return 1
    save_key(cfg, key)
    ui.ok("saved to {} (mode 600)".format(cfg.key_file))
    return 0


def cmd_check(cfg, args):
    ui.step("Setup check (hegemon-agent {})".format(VERSION))
    problems = []

    def row(good, label, detail="", hard=True):
        (ui.ok if good else (ui.fail if hard else ui.warn))(label + (": " + str(detail) if detail else ""))
        if not good and hard:
            problems.append(label)

    row(sys.version_info >= (3, 8), "Python " + sys.version.split()[0], sys.executable)
    for pkg, hard in (("pandas", True), ("numpy", True), ("openpyxl", False), ("h5py", False)):
        try:
            mod = __import__(pkg)
            row(True, pkg + " " + getattr(mod, "__version__", ""))
        except ImportError:
            row(False, pkg + " not installed", hard=hard)
    row(cfg.lab_script.is_file(), "lab script", cfg.lab_script)
    genomes = [n for n in ("Homo_sapiens.GRCh38.94.chr_patch_hapl_scaff.t.txt",
                           "Mus_musculus.GRCm38.94.chr_patch_hapl_scaff.t.txt") if (cfg.genome_dir / n).is_file()]
    row(bool(genomes), "lab genome tables", "{} ({} of 2)".format(cfg.genome_dir, len(genomes)), hard=False)
    try:
        cfg.runs_root.mkdir(parents=True, exist_ok=True)
        probe = cfg.runs_root / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        row(True, "runs folder", cfg.runs_root)
    except OSError as err:
        row(False, "runs folder not writable", err)
    row((cfg.site_root / "explore.conf").is_file() and (cfg.site_root / "index.html").is_file(),
        "Hegemon site files", cfg.site_root)
    try:
        page = net.get_text(cfg.site_url + "/explore.php", timeout=30, retries=0)
        row("select dataset" in page.lower(), "Hegemon page", cfg.site_url + "/explore.php", hard=False)
    except net.NetError as err:
        row(False, "Hegemon page unreachable", err, hard=False)
    ui.info("mode: " + MODE_TEXT[cfg.mode or "cloud"] + ("" if cfg.mode else " -- not chosen yet; hegemon mode"))
    key = load_key(cfg, prompt=False) if cfg.needs_key() else None
    if cfg.needs_key():
        row(bool(key), "OpenAI API key", "found" if key else "missing: run hegemon key")
    if key or not cfg.needs_key():
        llm = LLM(key, cfg)
        for model in dict.fromkeys((cfg.model, cfg.fast_model)):
            where = "Local model {} at {}".format(model, cfg.local_url) if cfg.is_local(model) else "OpenAI model " + model
            try:
                llm.ping(model)
                row(True, where, "answered a test request")
            except ProviderError as err:
                row(False, where, err)
    ui.step("Setup OK" if not problems else "Fix first: " + ", ".join(problems))
    return 0 if not problems else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hegemon", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd")

    def build_opts(p):
        p.add_argument("--platform", help="GPL to build when a GEO series has several")
        p.add_argument("--source", help="use files already downloaded to this folder")
        p.add_argument("--url", action="append", metavar="URL",
                       help="download this file (repeat for several); works for any dataset, any repository")
        p.add_argument("--script", help="start from this conversion script instead of asking the AI")
        p.add_argument("--attempts", type=int, help="conversion attempts (default 5)")
        p.add_argument("--yes", action="store_true", help="answer yes to size and publish questions")
        p.add_argument("--genes", metavar="A,B",
                       help="two genes the published page opens on (default: chosen for the topic, else housekeeping)")
        p.add_argument("--min-samples", type=int, default=None,
                       help="skip studies with fewer samples (default 20 for search/batch, 0 for build)")
        p.add_argument("--max-samples", type=int, default=None,
                       help="skip studies with more samples; the biggest take longest (default 250 for "
                            "search/batch, no limit for build; 0 = no limit)")
        p.add_argument("--no-publish", action="store_true", help="build only")

    def batch_opts(p):
        p.add_argument("--count", type=int, default=10,
                       help="how many of the best datasets for this search the page should hold (default 10); "
                            "with --more, how many to add")
        p.add_argument("--more", action="store_true",
                       help="add the next --count datasets down the ranking, instead of only what is new in the "
                            "best --count")
        p.add_argument("--label", help="the text for the single homepage link (default: the query)")
        p.add_argument("--retry-failed", action="store_true",
                       help="also try again the datasets that did not publish in an earlier run")
        p.add_argument("--rerank", action="store_true",
                       help="forget the ranking remembered for this search and rank every study again")

    s = sub.add_parser("search", help="search, pick, build, publish")
    s.add_argument("query", nargs="*")
    build_opts(s)
    b = sub.add_parser("build", help="build one accession")
    b.add_argument("accession")
    build_opts(b)
    ba = sub.add_parser("batch", help="process several datasets and put them all on one page (one shared key)")
    ba.add_argument("query", nargs="*", help="what to search for (optional when the key already exists)")
    ba.add_argument("--key", required=True, help="the Hegemon key all these datasets share, e.g. tregcd")
    batch_opts(ba)
    build_opts(ba)
    ag = sub.add_parser("agent", help="ask what to look for, then put the 10 best datasets on one Hegemon page")
    ag.add_argument("query", nargs="*", help="what to search for (asked when left out)")
    ag.add_argument("--key", help="the Hegemon key for the page (default: made from the words you type)")
    batch_opts(ag)
    build_opts(ag)
    sub.add_parser("install", help="make `hegemon` a command in your shell (run once)")

    p = sub.add_parser("publish", help="publish a finished build folder")
    p.add_argument("build")
    p.add_argument("--key", help="publish under this (shared) key")
    p.add_argument("--genes", metavar="A,B", help="two genes the page opens on")
    c = sub.add_parser("compare", help="compare a build with the lab's hand-made files")
    c.add_argument("agent")
    c.add_argument("lab")
    c.add_argument("--tol", type=float, default=1e-3)
    dg = sub.add_parser("diagnose", help="check the datasets already on your Hegemon site")
    dg.add_argument("--key", help="only the datasets under this key")
    dg.add_argument("--min-samples", type=int, default=20, help="flag datasets with fewer samples (default 20)")

    rl = sub.add_parser("relink", help="make each key's link open on a plot (genes every dataset has)")
    rl.add_argument("--key", help="only this key")
    rl.add_argument("--genes", metavar="A,B", help="use these two genes when every dataset has them")

    up = sub.add_parser("unpublish", help="take datasets off your Hegemon site (files are kept)")
    up.add_argument("sections", nargs="*", help="section names from explore.conf, as diagnose prints them")
    up.add_argument("--key", help="take off every dataset under this key")
    up.add_argument("--yes", action="store_true", help="do not ask for confirmation")

    sub.add_parser("mode", help="choose local, mixed or cloud models")
    sub.add_parser("check", help="check setup and the AI model(s)")
    sub.add_parser("key", help="save the OpenAI API key")
    args = parser.parse_args(argv)
    if getattr(args, "min_samples", None) is None:
        args.min_samples = 20 if args.cmd in ("search", "batch", "agent", None) else 0
    if getattr(args, "max_samples", None) is None:
        args.max_samples = 250 if args.cmd in ("search", "batch", "agent", None) else 0
    cfg = Config()
    if getattr(args, "attempts", None):
        cfg.attempts = args.attempts
    try:
        if args.cmd == "build":
            return cmd_build(cfg, args)
        return {"search": cmd_search, None: cmd_search, "batch": cmd_batch, "agent": cmd_agent, "mode": cmd_mode,
                "install": cmd_install, "publish": cmd_publish,
                "compare": cmd_compare, "check": cmd_check, "key": cmd_key,
                "diagnose": cmd_diagnose, "unpublish": cmd_unpublish, "relink": cmd_relink}[args.cmd](cfg, args)
    except KeyboardInterrupt:
        print("\n   Stopped with Ctrl+C. Partial files stay in the build folder.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

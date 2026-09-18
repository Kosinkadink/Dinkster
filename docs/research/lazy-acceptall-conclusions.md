# Lazy + arbitrary-input usage: ecosystem conclusions

Directly-referenceable conclusions of the 2026-07-29 feature-usage
census (user directive). Full data, scanner, spec, and per-pack
classifications live in the private
[Kosinkadink/dinkster-research](https://github.com/Kosinkadink/dinkster-research)
repository: `feature-usage/LAZY-ACCEPTALL.md` plus
`feature-usage/results/` at commit `8089cae` (scanner `scan_lazy.py` +
`LAZY-SCAN-SPEC.md` at `0cc62c2`). Method: mechanical detection over
fresh shallow clones of the 747-pack usage-weighted roster (per-pack
HEAD commits pinned), then CODE-READ classification of all 241 flagged
packs (578 usage rows) by three delegate threads, coordinator-verified.

## Conclusions

1. **All lazy usage is switch/if-else nodes.** 88 confirmed rows, all
   branch-selection. Split: 11.8 percent of user-weighted installs
   select via a document-visible widget (kjnodes, easy-use,
   impact-pack, crystools) - fully covered by a declarative
   document-time conditional with no lazy protocol; only 0.6 percent
   (plus core ComfySwitchNode) select via a computed upstream value
   and need runtime-value conditional evaluation.
2. **No pack uses multi-round imperative check_lazy_status.** Every
   use is "evaluate exactly the selected branch(es)".
3. **Arbitrary inputs are families, not chaos.** 18.1 percent of
   user-weighted installs use predictable frontend-JS autogrown
   member families (image_N, lora_N, in1..N) consumed via **kwargs -
   representable by the existing wire-15 prefix/names families; the
   remaining work is V1 compat lowering of JS-declared families, not
   schema capacity.
4. **User-named open inputs are document-carried.** 4.6 percent of
   users (dominated by rgthree DynamicContext, 18.1k installs) use
   names that are not a suffix family, but in every case the names
   come from document state (user labels, code variable names, field
   IDs, pasted source). A names-form family with document-defined
   member vocabulary covers the whole category.
5. **Zero uses of genuinely runtime-open kwargs in 747 packs.** The
   exclusion of general open-input support and the full lazy callback
   protocol stands on ecosystem-wide evidence.
6. **V1 dominates real usage** (207 of 234 confirmed non-false-positive
   rows); V3 confirmed uses concentrate in kjnodes, easy-use,
   xiser_nodes, and core.

Consumers: ROADMAP "Restore lazy/accept-all catalog nodes" entry
(re-weighted 2026-07-29 with these numbers); core-node scoping detail
in the coordinator workspace note
notes/deferral-estimates-switchnode-customcombo.md (station12).

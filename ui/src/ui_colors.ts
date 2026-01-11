import type { Assertion, SemanticGroup } from "./types";

function baseBadge(extra: string): string {
  return [
    "inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-semibold uppercase tracking-wide",
    extra,
  ].join(" ");
}

export function assertionBadgeClass(a: Assertion): string {
  switch (a) {
    case "PRESENT":
      return baseBadge("border-emerald-400/18 bg-emerald-600/14 text-emerald-100");
    case "ABSENT":
      return baseBadge("border-rose-400/18 bg-rose-600/14 text-rose-100");
    case "POSSIBLE":
      return baseBadge("border-amber-400/18 bg-amber-600/14 text-amber-100");
  }
}

export function groupBadgeClass(g: SemanticGroup): string {
  const extra = {
    DISO: "border-fuchsia-400/18 bg-fuchsia-600/14 text-fuchsia-100",
    CHEM: "border-cyan-400/18 bg-cyan-600/14 text-cyan-100",
    PROC: "border-amber-400/18 bg-amber-600/14 text-amber-100",
    ANAT: "border-emerald-400/18 bg-emerald-600/14 text-emerald-100",
    PHEN: "border-lime-400/18 bg-lime-600/14 text-lime-100",
    PHYS: "border-sky-400/18 bg-sky-600/14 text-sky-100",
    GENE: "border-indigo-400/18 bg-indigo-600/14 text-indigo-100",
    CONC: "border-slate-400/18 bg-slate-600/14 text-slate-100",
    ACTI: "border-violet-400/18 bg-violet-600/14 text-violet-100",
    DEVI: "border-teal-400/18 bg-teal-600/14 text-teal-100",
    GEOG: "border-blue-400/18 bg-blue-600/14 text-blue-100",
    LIVB: "border-pink-400/18 bg-pink-600/14 text-pink-100",
    OBJC: "border-slate-400/18 bg-slate-600/14 text-slate-100",
    OCCU: "border-slate-400/18 bg-slate-600/14 text-slate-100",
    ORGA: "border-slate-400/18 bg-slate-600/14 text-slate-100",
  }[g];
  return baseBadge(extra ?? "border-slate-700 bg-slate-800 text-slate-200");
}

export function groupMarkClass(g: SemanticGroup): string {
  switch (g) {
    case "DISO":
      return "bg-fuchsia-600/35 hover:bg-fuchsia-600/45";
    case "CHEM":
      return "bg-cyan-600/30 hover:bg-cyan-600/40";
    case "PROC":
      return "bg-amber-600/30 hover:bg-amber-600/40";
    case "ANAT":
      return "bg-emerald-600/25 hover:bg-emerald-600/35";
    case "PHEN":
      return "bg-lime-600/25 hover:bg-lime-600/35";
    case "PHYS":
      return "bg-sky-600/25 hover:bg-sky-600/35";
    case "GENE":
      return "bg-indigo-600/25 hover:bg-indigo-600/35";
    case "CONC":
      return "bg-slate-600/25 hover:bg-slate-600/35";
    default:
      return "bg-slate-600/25 hover:bg-slate-600/35";
  }
}

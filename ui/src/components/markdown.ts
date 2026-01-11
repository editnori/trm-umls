import { marked, type Tokens } from "marked";
import markedKatex from "marked-katex-extension";

function escapeHtml(input: string): string {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

const renderer = new marked.Renderer();

renderer.code = (token: Tokens.Code) => {
  const language = (token.lang || "").trim().toLowerCase();
  if (language === "mermaid") {
    return `<div class="mermaid">${escapeHtml(token.text)}</div>`;
  }
  return `<pre><code class="language-${escapeHtml(language)}">${escapeHtml(token.text)}</code></pre>`;
};

renderer.heading = (token: Tokens.Heading) => {
  const slug = slugify(token.text);
  const inner = marked.parseInline(token.text);
  return `<h${token.depth} id="${slug}">${inner}</h${token.depth}>`;
};

marked.setOptions({
  gfm: true,
  breaks: false,
  renderer,
});

marked.use(
  markedKatex({
    throwOnError: false,
    displayMode: true,
    output: "html",
    strict: "ignore",
  }),
);

export function renderMarkdown(source: string): string {
  return marked.parse(source) as string;
}

function slugify(input: string): string {
  return input
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-");
}

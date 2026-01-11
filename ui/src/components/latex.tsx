import katex from "katex";

export function Latex(props: { tex: string; display?: boolean; className?: string; title?: string }) {
  const html = katex.renderToString(props.tex, {
    displayMode: Boolean(props.display),
    throwOnError: false,
    strict: "ignore",
    trust: false,
    output: "html",
  });
  return (
    <span
      className={props.className}
      title={props.title}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}


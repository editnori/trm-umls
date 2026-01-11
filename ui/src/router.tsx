import { useEffect, useState } from "react";
import App from "./App";
import Methodology from "./Methodology";

const THEME_KEY = "trm_umls_theme";

type Page = "app" | "methodology";

function pageFromHash(): Page {
  const hash = window.location.hash.slice(1);
  return hash.startsWith("methodology") ? "methodology" : "app";
}

export default function Router() {
  const [page, setPage] = useState<Page>(() => pageFromHash());

  useEffect(() => {
    const saved = window.localStorage.getItem(THEME_KEY);
    const theme = saved === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", theme);

    const handleHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", handleHash);
    return () => window.removeEventListener("hashchange", handleHash);
  }, []);

  if (page === "methodology") return <Methodology />;
  return <App />;
}

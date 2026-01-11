# trm-umls ui

This is a local viewer for `trm_umls/` that:

- uploads one or more note files (kept in memory)
- calls the local api to extract spans → CUIs
- renders highlighted mentions plus a table with filters

## run

Start the api (loads model + index once):

```bash
cd /path/to/Medgemma
python3 -m trm_umls.api
```

Start the ui:

```bash
cd ui
bun install
bun run dev
```

Open `http://localhost:5173`.

## config

By default the ui calls `http://127.0.0.1:8000`.

You can override with:

```bash
VITE_API_BASE=http://127.0.0.1:8000 bun run dev
```

## privacy

This is intended to run locally. Do not point it at a remote api if your notes contain PHI.

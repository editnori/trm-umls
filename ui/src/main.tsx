import { StrictMode, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import Methodology from './Methodology.tsx'

function Router() {
  const [page, setPage] = useState(() => {
    const hash = window.location.hash.slice(1);
    return hash === 'methodology' ? 'methodology' : 'app';
  });

  useEffect(() => {
    const handleHash = () => {
      const hash = window.location.hash.slice(1);
      setPage(hash === 'methodology' ? 'methodology' : 'app');
    };
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);

  if (page === 'methodology') {
    return <Methodology />;
  }
  return <App />;
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Router />
  </StrictMode>,
)

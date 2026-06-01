import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

// Note: intentionally not wrapped in <StrictMode>. Its dev-only double
// mount/unmount would create and tear down the WebGL context twice, which
// is fragile for the GPGPU particle pipeline.
createRoot(document.getElementById("root")!).render(<App />);

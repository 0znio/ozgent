// Applied before the first paint, so a saved theme never flashes the other one.
// Loaded as a plain (blocking) script in <head> for exactly that reason.
try {
  var saved = localStorage.getItem("ozgent-theme");
  if (saved === "light" || saved === "dark") {
    document.documentElement.setAttribute("data-theme", saved);
  }
} catch (e) { /* private browsing; fall back to the system preference */ }

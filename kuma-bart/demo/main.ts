import "../src/player/KumaPlayer.js";

const player = document.querySelector<HTMLElement>("#player")!;
const modelSelect = document.querySelector<HTMLSelectElement>("#model-select")!;
const vanillaBtn = document.querySelector<HTMLButtonElement>("#mode-vanilla")!;
const debugBtn = document.querySelector<HTMLButtonElement>("#mode-debug")!;

modelSelect.addEventListener("change", () => {
  if (!modelSelect.value) return;
  player.setAttribute("src", modelSelect.value);
});

function setMode(debug: boolean): void {
  player.toggleAttribute("vanilla", !debug);
  player.toggleAttribute("debug", debug);
  vanillaBtn.setAttribute("aria-pressed", String(!debug));
  debugBtn.setAttribute("aria-pressed", String(debug));
}

vanillaBtn.addEventListener("click", () => setMode(false));
debugBtn.addEventListener("click", () => setMode(true));

async function populateArtifacts(): Promise<void> {
  let files: string[] = [];
  try {
    const res = await fetch("/api/artifacts");
    files = (await res.json()) as string[];
  } catch {
    // dev server not available or no artifacts dir — leave list empty
  }

  modelSelect.innerHTML = "";

  if (files.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(no .iph files found in artifacts/)";
    modelSelect.appendChild(opt);
    return;
  }

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "— select a model —";
  modelSelect.appendChild(placeholder);

  for (const name of files) {
    const opt = document.createElement("option");
    opt.value = `/artifacts/${name}`;
    opt.textContent = name;
    modelSelect.appendChild(opt);
  }
}

void populateArtifacts();

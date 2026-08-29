import { useMemo, useRef, useState } from "react";
import {
  Activity,
  BrainCircuit,
  Check,
  CircleHelp,
  Download,
  FileImage,
  Gauge,
  Info,
  Layers3,
  LayoutDashboard,
  Moon,
  RotateCcw,
  ScanLine,
  Settings2,
  ShieldCheck,
  Sun,
  Upload,
  XCircle,
} from "lucide-react";
import { jsPDF } from "jspdf";
import { trpc } from "@/lib/trpc";
import { useTheme } from "@/contexts/ThemeContext";
import {
  CLASS_NAMES as CLASSES,
  LOW_CONFIDENCE_THRESHOLD,
  displayLabel,
  isNoTumor,
  resultLabel,
} from "@shared/model";

const STORAGE_ATLAS = "/storage/atlas-linework_e1bc9e39.png";

// Default current-scan image from the project assets
const DEFAULT_PLACEHOLDER_SVG = new URL("../../../brainimg.png", import.meta.url)
  .href;

// Upload zone illustration with UI elements
const UPLOAD_ILLUSTRATION_SVG = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 280'%3E%3Cdefs%3E%3Cstyle%3E.upload-glow%7Bfill:none;stroke:%232d9b94;stroke-width:2;opacity:0.4;%7D.upload-icon%7Bfill:%232d9b94;%7D%3C/style%3E%3C/defs%3E%3Crect x='60' y='40' width='180' height='140' rx='8' class='upload-glow'/%3E%3Cpath d='M 150 80 L 150 130 M 130 110 L 150 90 L 170 110' class='upload-icon' stroke-linecap='round' stroke-linejoin='round' stroke-width='3' fill='none'/%3E%3Ccircle cx='150' cy='180' r='35' fill='none' stroke='%235aa3e8' stroke-width='2' opacity='0.3'/%3E%3Ccircle cx='150' cy='180' r='28' fill='none' stroke='%232d9b94' stroke-width='1.5' opacity='0.2' stroke-dasharray='5,5'/%3E%3Ctext x='150' y='235' font-family='Arial,sans-serif' font-size='13' fill='%234a90e2' text-anchor='middle' font-weight='bold' opacity='0.7'%3EDROP OR CLICK TO UPLOAD%3C/text%3E%3C/svg%3E`;

type AnalysisResult = {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  originalImage: string;
  gradcam: string;
  imageWidth: number;
  imageHeight: number;
};
type BatchCase = {
  id: string;
  name: string;
  previewUrl: string;
  status: "queued" | "processing" | "complete" | "error";
  result?: AnalysisResult;
  error?: string;
};
function readFile(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Could not read this file."));
    reader.readAsDataURL(file);
  });
}
function imageDataUrl(base64: string) {
  return `data:image/png;base64,${base64}`;
}
function isReadableImageDataUrl(dataUrl: string) {
  const payload = dataUrl.split(",", 2)[1] ?? "";
  return payload.startsWith("/9j/") || payload.startsWith("iVBORw0KGgo");
}

export default function Home() {
  const { theme, toggleTheme } = useTheme();
  const [cases, setCases] = useState<BatchCase[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedId, setSelectedId] = useState<string | undefined>();
  const [layer, setLayer] = useState<"original" | "gradcam">("original");
  const inputRef = useRef<HTMLInputElement>(null);
  const analyzeMutation = trpc.analyze.useMutation();
  const completed = cases.filter(item => item.result);
  const selected =
    cases.find(item => item.id === selectedId) ??
    completed[completed.length - 1] ??
    cases[0];
  const result = selected?.result;
  const isBusy = cases.some(item => item.status === "processing");
  const tumorCount = completed.filter(
    item => item.result && !isNoTumor(item.result.prediction)
  ).length;
  const updateCase = (id: string, patch: Partial<BatchCase>) =>
    setCases(current =>
      current.map(item => (item.id === id ? { ...item, ...patch } : item))
    );
  const handleFiles = async (fileList?: FileList | File[]) => {
    const files = Array.from(fileList ?? []).filter(file =>
      ["image/png", "image/jpeg"].includes(file.type)
    );
    if (!files.length) return;
    const newCases = files.map(file => ({
      id: `${file.name}-${file.lastModified}-${Math.random()}`,
      name: file.name,
      previewUrl: "",
      status: "queued" as const,
    }));
    setCases(current => [...current, ...newCases]);
    setSelectedId(newCases[0]?.id);
    setLayer("original");
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      const item = newCases[index];
      if (!item) continue;
      try {
        const dataUrl = await readFile(file);
        updateCase(item.id, { previewUrl: dataUrl, status: "processing" });
        if (!isReadableImageDataUrl(dataUrl))
          throw new Error(
            "The uploaded file is not a readable PNG or JPEG image."
          );
        const response = (await analyzeMutation.mutateAsync({
          imageDataUrl: dataUrl,
          mimeType: file.type,
        })) as AnalysisResult;
        updateCase(item.id, { result: response, status: "complete" });
      } catch (error) {
        updateCase(item.id, {
          status: "error",
          error:
            error instanceof Error
              ? error.message
              : "The model could not analyze this image.",
        });
      }
    }
  };
  const reset = () => {
    setCases([]);
    setSelectedId(undefined);
    analyzeMutation.reset();
    if (inputRef.current) inputRef.current.value = "";
  };
  const exportPdf = () => {
    if (!completed.length) return;
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    completed.forEach((item, index) => {
      if (index) doc.addPage();
      const current = item.result!;
      const label = resultLabel(current.prediction, current.confidence);
      doc.setTextColor(19, 29, 43);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.text("NEUROLENS / BATCH MODEL REPORT", 42, 45);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(9);
      doc.setTextColor(90, 103, 120);
      doc.text(
        `Case ${index + 1} of ${completed.length} · ${item.name}`,
        42,
        62
      );
      doc.setTextColor(19, 29, 43);
      doc.setFontSize(24);
      doc.text(label, 42, 105);
      doc.setFontSize(12);
      doc.text(`Model confidence: ${current.confidence}%`, 42, 128);
      doc.setFontSize(10);
      doc.setTextColor(90, 103, 120);
      doc.text("Probability distribution", 42, 160);
      let y = 180;
      CLASSES.forEach(className => {
        doc.text(
          `${displayLabel(className)}: ${current.probabilities[className] ?? 0}%`,
          42,
          y
        );
        y += 16;
      });
      try {
        doc.addImage(
          imageDataUrl(current.originalImage),
          "PNG",
          42,
          270,
          220,
          290
        );
        doc.addImage(imageDataUrl(current.gradcam), "PNG", 300, 270, 220, 290);
      } catch {
        doc.setTextColor(150, 70, 50);
        doc.text("Image preview unavailable in this report.", 42, 590);
      }
      doc.setFontSize(8);
      doc.setTextColor(110, 120, 120);
      doc.text(
        "For research demonstration only. Model confidence is not clinical accuracy or a diagnosis.",
        42,
        pageHeight - 35
      );
    });
    doc.save(
      `neurolens-batch-report-${new Date().toISOString().slice(0, 10)}.pdf`
    );
  };
  const activeImage = result
    ? layer === "original"
      ? imageDataUrl(result.originalImage)
      : imageDataUrl(result.gradcam)
    : selected?.previewUrl || DEFAULT_PLACEHOLDER_SVG;
  const isInconclusive = Boolean(
    result && result.confidence < LOW_CONFIDENCE_THRESHOLD
  );
  const selectedLabel = result
    ? resultLabel(result.prediction, result.confidence)
    : selected?.status === "error"
      ? "Analysis failed"
      : selected?.status === "processing"
        ? "Analyzing scan…"
        : "Current scan";
  const distribution = useMemo(
    () =>
      result
        ? CLASSES.map(name => ({
            name,
            value: result.probabilities[name] ?? 0,
          }))
        : [],
    [result]
  );
  return (
    <div className="dashboard-shell">
      <aside className="sidebar">
        <a className="brand dark-brand" href="#dashboard">
          <span className="brand-mark">
            <BrainCircuit size={22} />
          </span>
          <span>
            <strong>
              Neuro<span>Scan</span>
            </strong>
            <small>AI Brain MRI Analysis</small>
          </span>
        </a>
        <nav className="side-nav">
          <a className="active" href="#dashboard">
            <LayoutDashboard size={18} /> Dashboard
          </a>
          <a href="#upload">
            <Upload size={18} /> Scan MRI
          </a>
          <a href="#results">
            <Activity size={18} /> Test Results
          </a>
          <a href="#model">
            <Info size={18} /> About Model
          </a>
          <a href="#help">
            <CircleHelp size={18} /> How It Works
          </a>
        </nav>
        <div className="privacy-card">
          <ShieldCheck size={28} />
          <strong>Your Data is Safe</strong>
          <p>We don't store your scans. All analysis is private and secure.</p>
        </div>
      </aside>
      <main className="dashboard-main" id="dashboard">
        <header className="dashboard-topbar">
          <div className="mobile-brand">
            NEURO<span>SCAN</span>
          </div>
          <div className="topbar-spacer" />
          <button
            className="theme-toggle"
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            onClick={() => toggleTheme?.()}
          >
            <Sun
              size={15}
              className={theme === "light" ? "theme-active" : ""}
            />
            <Moon
              size={15}
              className={theme === "dark" ? "theme-active" : ""}
            />
          </button>
          <button
            className="scan-new-button"
            onClick={() => inputRef.current?.click()}
          >
            <Upload size={16} /> Scan New MRI
          </button>
        </header>
        <section className="dashboard-content">
          <div className="dashboard-heading">
            <div>
              <p className="dashboard-kicker">01 / ANALYSIS CONSOLE</p>
              <h1>
                Brain MRI <span>Dashboard</span>
              </h1>
              <p>
                Review batch scans, model confidence, and class distributions in
                one clinical workspace.
              </p>
            </div>
            <div className="live-pill">
              <i /> MODEL ONLINE
            </div>
          </div>
          <section className="kpi-grid">
            <div className="kpi-card">
              <span className="kpi-icon blue">
                <ScanLine size={20} />
              </span>
              <div>
                <small>Scans Analyzed</small>
                <strong>{completed.length}</strong>
                <em>Total scans in run</em>
              </div>
            </div>
            <div className="kpi-card">
              <span className="kpi-icon pink">
                <BrainCircuit size={20} />
              </span>
              <div>
                <small>Tumor Detected</small>
                <strong>{tumorCount}</strong>
                <em>Based on completed scans</em>
              </div>
            </div>
          </section>
          <section className="workspace-grid">
            <div className="scan-workspace panel">
              <div className="panel-header">
                <div>
                  <small>CURRENT SCAN</small>
                  <h2>{selected?.name ?? "No scan selected"}</h2>
                </div>
                <div className="layer-tabs">
                  <button
                    className={layer === "original" ? "active" : ""}
                    onClick={() => setLayer("original")}
                  >
                    Original
                  </button>
                  <button
                    className={layer === "gradcam" ? "active" : ""}
                    onClick={() => setLayer("gradcam")}
                    disabled={!result}
                  >
                    Grad-CAM
                  </button>
                </div>
              </div>
              <div className="scan-viewport">
                {activeImage ? (
                  <img
                    src={activeImage}
                    alt={
                      layer === "gradcam"
                        ? "Grad-CAM visualization"
                        : "Uploaded MRI scan"
                    }
                  />
                ) : null}
                {!result && !selected?.previewUrl && (
                  <div className="viewport-empty">
                    <ScanLine size={34} />
                    <span>Upload an MRI scan to begin</span>
                  </div>
                )}
                <span className="orientation left">R</span>
                <span className="orientation right">L</span>
                <span className="viewport-tag">
                  {layer === "gradcam" ? "ATTENTION MAP" : "AXIAL VIEW"}
                </span>
              </div>
              {result && (
                <div className="scan-meta">
                  <span>
                    <FileImage size={14} /> {result.imageWidth} ×{" "}
                    {result.imageHeight}px
                  </span>
                  <span>
                    <Layers3 size={14} />{" "}
                    {layer === "gradcam" ? "Grad-CAM overlay" : "Source image"}
                  </span>
                  <span className="meta-ready">
                    <Check size={14} /> Result ready
                  </span>
                </div>
              )}
              <div
                className={`upload-dropzone ${isDragging ? "is-dragging" : ""}`}
                id="upload"
                onDragOver={event => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={event => {
                  event.preventDefault();
                  setIsDragging(false);
                  void handleFiles(event.dataTransfer.files);
                }}
              >
                <input
                  id="mri-upload"
                  ref={inputRef}
                  type="file"
                  className="file-input-dashboard"
                  multiple
                  accept="image/png,image/jpeg"
                  aria-label="Upload multiple MRI images"
                  onChange={event => {
                    void handleFiles(event.target.files ?? undefined);
                  }}
                />
                <div className="upload-content">
                  <img
                    src={UPLOAD_ILLUSTRATION_SVG}
                    alt="Upload illustration"
                    className="upload-illustration"
                  />
                  <div className="upload-text">
                    <strong>Add MRI Scan to Analyze</strong>
                    <span>Drag and drop here or click to browse</span>
                    <small>JPG, JPEG, PNG • Max 12 MB each</small>
                  </div>
                  <button
                    className="upload-browse-button"
                    onClick={() => inputRef.current?.click()}
                  >
                    <Upload size={16} /> Choose File
                  </button>
                </div>
              </div>
            </div>
            <aside className="prediction-panel panel" id="results">
              <div className="panel-header">
                <div>
                  <small>PREDICTION RESULT</small>
                  <h2>{result ? "Analysis complete" : "Awaiting scan"}</h2>
                </div>
                <Settings2 size={17} className="panel-muted" />
              </div>
              {result ? (
                <>
                  <div
                    className={`prediction-hero ${isNoTumor(result.prediction) ? "no-tumor" : ""}`}
                  >
                    <span className="prediction-symbol">
                      <BrainCircuit size={25} />
                    </span>
                    <div>
                      <h3>{selectedLabel}</h3>
                      <p>
                        {isInconclusive
                          ? "The model confidence is below the reporting threshold; review the full distribution and seek professional interpretation."
                          : isNoTumor(result.prediction)
                            ? "No tumor signal was identified by the model."
                            : "The model identified a tumor class in this scan."}
                      </p>
                    </div>
                  </div>
                  <div className="confidence-box">
                    <div>
                      <span>Confidence Score</span>
                      <strong>{result.confidence}%</strong>
                    </div>
                    <div className="confidence-track">
                      <i style={{ width: `${result.confidence}%` }} />
                    </div>
                    <div className="scale">
                      <span>0%</span>
                      <span>100%</span>
                    </div>
                  </div>
                  <div className="details-list">
                    <h3>Analysis Details</h3>
                    <div>
                      <span>
                        <Settings2 size={14} /> Model used
                      </span>
                      <strong>EfficientNetB0</strong>
                    </div>
                    <div>
                      <span>
                        <Layers3 size={14} /> Classes
                      </span>
                      <strong>4 signals</strong>
                    </div>
                    <div>
                      <span>
                        <ScanLine size={14} /> Resolution
                      </span>
                      <strong>
                        {result.imageWidth} × {result.imageHeight}
                      </strong>
                    </div>
                    <div>
                      <span>
                        <Activity size={14} /> Confidence type
                      </span>
                      <strong>Softmax score</strong>
                    </div>
                  </div>
                  <div className="distribution-box">
                    <div className="distribution-heading">
                      <h3>Probability Distribution</h3>
                      <span>100% total</span>
                    </div>
                    {distribution.map(item => (
                      <div
                        className={`dist-row ${item.name === result.prediction ? "selected" : ""}`}
                        key={item.name}
                      >
                        <span>{displayLabel(item.name)}</span>
                        <div>
                          <i style={{ width: `${item.value}%` }} />
                        </div>
                        <strong>{item.value}%</strong>
                      </div>
                    ))}
                  </div>
                  <button
                    className="export-button"
                    onClick={exportPdf}
                    disabled={!completed.length}
                  >
                    <Download size={16} /> Download Report
                  </button>
                </>
              ) : (
                <div className="prediction-empty">
                  <Gauge size={30} />
                  <p>
                    Upload one or more scans to view the detected class,
                    confidence, and complete probability distribution.
                  </p>
                </div>
              )}
              <button className="reset-dashboard" onClick={reset}>
                <RotateCcw size={14} /> Analyze Another Scan
              </button>
            </aside>
          </section>
          {cases.length > 0 && (
            <section className="case-strip">
              <div className="case-strip-head">
                <div>
                  <small>BATCH QUEUE</small>
                  <h2>
                    {completed.length} of {cases.length} analyzed
                  </h2>
                </div>
                <button onClick={reset}>
                  <XCircle size={14} /> Clear batch
                </button>
              </div>
              <div className="case-grid">
                {cases.map((item, index) => (
                  <button
                    className={`case-card ${item.id === selected?.id ? "selected" : ""} ${item.status}`}
                    key={item.id}
                    onClick={() => setSelectedId(item.id)}
                  >
                    <span>CASE {String(index + 1).padStart(2, "0")}</span>
                    {item.previewUrl && (
                      <img src={item.previewUrl} alt={item.name} />
                    )}
                    <div>
                      <strong>{item.name}</strong>
                      <small>
                        {item.status === "complete" && item.result
                          ? `${resultLabel(item.result.prediction, item.result.confidence)} · ${item.result.confidence}%`
                          : item.status === "error"
                            ? item.error
                            : item.status === "processing"
                              ? "Analyzing…"
                              : "Queued"}
                      </small>
                    </div>
                    <i>
                      {item.status === "complete" ? (
                        <Check size={13} />
                      ) : item.status === "error" ? (
                        <XCircle size={13} />
                      ) : (
                        <Activity size={13} />
                      )}
                    </i>
                  </button>
                ))}
              </div>
            </section>
          )}
          <section className="disclaimer validation-alert">
            <ShieldCheck size={18} />
            <div>
              <strong>Model validation status</strong>
              <span>
                The supplied artifact is not validated for clinical use and has
                shown inconsistent class predictions on the provided labeled
                examples. Low-confidence results are marked inconclusive; do not
                use this tool for medical decisions.
              </span>
            </div>
          </section>
          <section className="info-row">
            <article id="model" className="info-card">
              <div className="info-card-icon">
                <Settings2 size={17} />
              </div>
              <div>
                <p className="dashboard-kicker">ABOUT MODEL</p>
                <h3>EfficientNetB0 · 4 classes</h3>
                <p>
                  Images are resized to 224 × 224 pixels; the embedded
                  EfficientNet preprocessing handles input scaling. Each score
                  is a model confidence, not clinical accuracy.
                </p>
              </div>
            </article>
            <article id="help" className="info-card">
              <div className="info-card-icon">
                <CircleHelp size={17} />
              </div>
              <div>
                <p className="dashboard-kicker">HOW IT WORKS</p>
                <h3>Upload → analyze → review</h3>
                <p>
                  Select one or more scans, inspect each predicted class and
                  distribution, then export the completed cases as a PDF report.
                </p>
              </div>
            </article>
          </section>
        </section>
        <footer className="dashboard-footer">
          NEUROSCAN / BRAIN TUMOR MODEL RESULTS <span>v2.1 · BATCH READY</span>
        </footer>
      </main>
    </div>
  );
}

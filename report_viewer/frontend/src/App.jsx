import React, { useEffect, useState } from "react";
import ImagePair from "./ImagePair";
import Report from "./Report";
import "../index.css";

function App() {
  const [items, setItems] = useState([]);
  const [index, setIndex] = useState(0);

  const [modelName, setModelName] = useState("");
  const [modelVersion, setModelVersion] = useState(0);

  useEffect(() => {
    fetch("http://localhost:8000/items")
      .then(res => res.json())
      .then(setItems)
      .catch(err => console.error(err));
  }, []);

  useEffect(() => {
  fetch("http://localhost:8000/model_name")
    .then(res => res.json())
    .then(data => setModelName(data.name));
  }, []);

  if (items.length === 0) {
    return (
      <div style={{ padding: 24, color: "#e5e7eb" }}>
        Initializing dashboard…
      </div>
    );
  }

  const id = items[index];

  const next = () => setIndex(i => (i + 1) % items.length);
  const prev = () => setIndex(i => (i - 1 + items.length) % items.length);

    const toggle = () => {
      fetch("http://localhost:8000/toggle", { method: "POST" })
        .then(() => fetch("http://localhost:8000/items"))
        .then(res => res.json())
        .then(newItems => {
          setItems(newItems);

          setIndex(i => {
            if (i >= newItems.length) return newItems.length - 1;
            return i;
          });

          setModelVersion(v => v + 1);
        });

      fetch("http://localhost:8000/model_name")
        .then(res => res.json())
        .then(data => setModelName(data.name));
    };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "radial-gradient(circle at top, #111827 0, #050608 55%)",
        color: "var(--text-main)"
      }}
    >
      <header
        style={{
          borderBottom: "1px solid var(--border)",
          padding: "12px 24px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          backgroundColor: "rgba(5, 6, 8, 0.9)",
          backdropFilter: "blur(8px)"
        }}
      >
        <div>
          <div style={{ fontSize: 18, letterSpacing: 1 }}>
            Object Detection on VisDrone Dataset
          </div>
          <div
            style={{
              fontSize: 11,
              color: "var(--text-muted)",
              textTransform: "uppercase",
              letterSpacing: 2
            }}
          >
            Detection Model Evaluation
          </div>
        </div>

        <div
          style={{
            fontSize: 11,
            color: "var(--accent)",
            textTransform: "uppercase",
            letterSpacing: 2
          }}
        >
        </div>
        <button
          onClick={toggle}
          style={{
            padding: "6px 12px",
            backgroundColor: "var(--panel-bg)",
            border: "1px solid var(--border)",
            color: "var(--text-main)",
            fontSize: 12,
            textTransform: "uppercase",
            letterSpacing: 1,
            cursor: "pointer"
          }}
        >
          {modelName}
        </button>
      </header>
      <main
        style={{
          padding: "24px",
          display: "grid",
          gridTemplateColumns: "2fr 1fr",
          gap: "16px"
        }}
      >
      <section>
          <div
            style={{
              backgroundColor: "var(--panel-bg)",
              borderRadius: 8,
              border: "1px solid var(--border)",
              padding: 16,
              boxShadow: "0 0 24px rgba(0,0,0,0.6)"
            }}
          >
            {/* Navigation + ID display */}
            <div
              style={{
                marginBottom: 12,
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between"
              }}
            >
              <div
                style={{
                  fontSize: 12,
                  color: "var(--text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: 2
                }}
              >
                Test Example: <span style={{ color: "var(--accent)" }}>{id}</span>
              </div>

              <div style={{ display: "flex", gap: 8 }}>
                <button
                  onClick={prev}
                  style={{
                    padding: "6px 12px",
                    backgroundColor: "var(--panel-bg)",
                    border: "1px solid var(--border)",
                    color: "var(--text-main)",
                    fontSize: 12,
                    textTransform: "uppercase",
                    letterSpacing: 1,
                    cursor: "pointer"
                  }}
                >
                  Previous
                </button>

                <button
                  onClick={next}
                  style={{
                    padding: "6px 12px",
                    backgroundColor: "var(--panel-bg)",
                    border: "1px solid var(--border)",
                    color: "var(--text-main)",
                    fontSize: 12,
                    textTransform: "uppercase",
                    letterSpacing: 1,
                    cursor: "pointer"
                  }}
                >
                  Next
                </button>
              </div>
            </div>
            <ImagePair id={id} modelVersion={modelVersion} />

            <div style={{ marginTop: 20 }}>
              <Report id={id} modelVersion={modelVersion} />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;

#!/usr/bin/env python3
"""
run_ui.py

A unified, zero-dependency Python script that:
1. Loads the pre-trained Llama-style decoder model (default: Run 5, the champion run).
2. Loads the SentencePiece tokenizer (default: Unigram v16k).
3. Spawns a lightweight HTTP server on localhost:8000 serving a premium, responsive
   dark-mode web interface to generate and experiment with Sinhala text.

GLOBAL VARIABLES: Change these to direct the server to load a different tokenizer or model.
"""

import http.server
import json
import os
import sys
import time
import urllib.parse
from http import HTTPStatus

# =============================================================================
# GLOBAL CONFIGURATION - Edit these to load different checkpoint models/tokenizers
# =============================================================================
MODEL_PATH = "Models/run5"
TOKENIZER_PATH = "Tokenizers/tokenizer/unigram_v16000_l16_i3.model"
PORT = 8000

# =============================================================================
# GLOBAL STATE & INITIALIZATION
# =============================================================================
model = None
tokenizer = None
device = "cpu"
model_load_time_s = 0.0
total_params = 0

def load_model_and_tokenizer():
    global model, tokenizer, device, model_load_time_s, total_params
    print("=" * 80)
    print("SINHALA LLM INFERENCE SERVER INITIALIZATION")
    print("=" * 80)
    
    # 1. Load Tokenizer
    print(f"[*] Step 1: Loading SentencePiece tokenizer from: {TOKENIZER_PATH} ...")
    if not os.path.exists(TOKENIZER_PATH):
        print(f"ERROR: Tokenizer file not found at '{TOKENIZER_PATH}'!")
        print("Please verify the path or copy your trained tokenizer .model file.")
        sys.exit(1)
        
    try:
        import sentencepiece as spm
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(TOKENIZER_PATH)
        print(f"[+] Tokenizer loaded. Vocab size: {tokenizer.get_piece_size():,}")
    except ImportError:
        print("ERROR: 'sentencepiece' package not installed. Run: pip install sentencepiece")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading tokenizer: {e}")
        sys.exit(1)

    # 2. Load Model
    print(f"[*] Step 2: Loading Llama causal LM from folder: {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model checkpoint directory not found at '{MODEL_PATH}'!")
        print("Confirm that training has completed and generated files saved.")
        sys.exit(1)

    try:
        import torch
        from transformers import LlamaForCausalLM
        
        start_time = time.time()
        
        # Check CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Target hardware device: {device.upper()}")
        
        model = LlamaForCausalLM.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()  # Set to evaluation mode for inference/generation
        
        model_load_time_s = round(time.time() - start_time, 2)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"[+] Model loaded successfully in {model_load_time_s}s!")
        print(f"[+] Total Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
        print(f"[+] Device in use: {device.upper()}")
        print("=" * 80)
    except ImportError:
        print("ERROR: Requirements missing. Run: pip install torch transformers")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading model weights: {e}")
        sys.exit(1)

# =============================================================================
# HTML FRONTEND TEMPLATE (Embedded Single Page App)
# =============================================================================
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sinhala GPT Playground</title>
    <!-- Modern Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Sinhala:wght@300;400;500;600;700&family=Outfit:wght@400;500;700&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --bg-color-active: #0c071d;
            --panel-bg: rgba(22, 17, 43, 0.45);
            --border-glow: rgba(147, 51, 234, 0.25);
            --accent-primary: #a855f7;
            --accent-secondary: #ec4899;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --font-main: 'Inter', 'Noto Sans Sinhala', sans-serif;
            --font-title: 'Outfit', sans-serif;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background: linear-gradient(135deg, #070311 0%, #12092f 50%, #06020c 100%);
            color: var(--text-main);
            font-family: var(--font-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            overflow-x: hidden;
        }

        header {
            width: 100%;
            max-width: 1200px;
            padding: 2.5rem 1.5rem 1.5rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo-section h1 {
            font-family: var(--font-title);
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(to right, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .logo-section p {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }

        .container {
            width: 100%;
            max-width: 1200px;
            padding: 0 1.5rem 3rem 1.5rem;
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 1.5rem;
            flex-grow: 1;
        }

        @media (max-width: 900px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        /* Glassmorphism Card Style */
        .glass-card {
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-glow);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: all 0.3s ease;
        }

        .glass-card:hover {
            border-color: rgba(236, 72, 153, 0.25);
            box-shadow: 0 8px 32px 0 rgba(147, 51, 234, 0.1);
        }

        /* Left configuration panel */
        .config-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            height: fit-content;
        }

        .config-title {
            font-family: var(--font-title);
            font-size: 1.2rem;
            font-weight: 600;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 0.75rem;
            color: #d8b4fe;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
        }

        .form-group label span.val {
            color: var(--accent-primary);
            font-weight: 600;
            font-family: monospace;
        }

        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--accent-primary);
            cursor: pointer;
            transition: transform 0.1s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
            background: var(--accent-secondary);
        }

        /* Model status block */
        .status-block {
            margin-top: 1rem;
            font-size: 0.8rem;
            display: flex;
            flex-direction: column;
            gap: 0.50rem;
            background: rgba(0, 0, 0, 0.2);
            padding: 0.8rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .status-item {
            display: flex;
            justify-content: space-between;
        }

        .status-item span.label {
            color: var(--text-muted);
        }

        .status-item span.value {
            font-family: monospace;
            color: #c084fc;
        }

        /* Generation Panel */
        .workspace-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .input-area-card {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        textarea {
            width: 100%;
            height: 140px;
            background: rgba(10, 5, 25, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            color: var(--text-main);
            padding: 1rem;
            font-size: 1rem;
            font-family: var(--font-main);
            resize: vertical;
            outline: none;
            transition: all 0.3s;
        }

        textarea:focus {
            border-color: var(--accent-primary);
            background: rgba(10, 5, 25, 0.6);
            box-shadow: 0 0 10px rgba(168, 85, 247, 0.15);
        }

        .btn-row {
            display: flex;
            justify-content: flex-end;
        }

        button.gen-btn {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.85rem 2rem;
            font-family: var(--font-title);
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 4px 14px rgba(236, 72, 153, 0.3);
            transition: all 0.3s;
        }

        button.gen-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(168, 85, 247, 0.4);
        }

        button.gen-btn:active:not(:disabled) {
            transform: translateY(0);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            box-shadow: none;
        }

        /* Output display */
        .output-card {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            min-height: 250px;
        }

        .output-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 0.75rem;
        }

        .output-header h2 {
            font-family: var(--font-title);
            font-size: 1.2rem;
            font-weight: 600;
            color: #d8b4fe;
        }

        .copy-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            color: var(--text-main);
            padding: 0.4rem 0.8rem;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .copy-btn:hover {
            background: rgba(255, 255, 255, 0.15);
            border-color: var(--accent-primary);
        }

        .output-display {
            background: rgba(10, 5, 25, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 1.2rem;
            font-size: 1.1rem;
            line-height: 1.6;
            white-space: pre-wrap;
            flex-grow: 1;
            min-height: 180px;
            overflow-y: auto;
            position: relative;
            color: #e5e7eb;
        }

        .placeholder-text {
            color: rgba(255, 255, 255, 0.2);
            font-style: italic;
        }

        /* Loading spinner */
        .spinner {
            display: none;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Dynamic Cursor blink for typing effect */
        .cursor {
            display: inline-block;
            width: 6px;
            height: 1.1rem;
            background-color: var(--accent-primary);
            margin-left: 2px;
            animation: blink 0.8s infinite;
            vertical-align: middle;
        }

        @keyframes blink {
            50% { opacity: 0; }
        }
    </style>
</head>
<body>

    <header>
        <div class="logo-section">
            <h1>Sinhala GPT Playground 🇱🇰</h1>
            <p>Interactive web UI to test and generate text from your trained Sinhala Llama model</p>
        </div>
    </header>

    <main class="container">
        <!-- Configuration Side Panel -->
        <section class="config-panel glass-card">
            <h2 class="config-title">Generation Settings</h2>
            
            <div class="form-group">
                <label for="temperature">Temperature <span class="val" id="temp-val">0.70</span></label>
                <input type="range" id="temperature" min="0.10" max="2.00" step="0.05" value="0.70">
            </div>

            <div class="form-group">
                <label for="max_tokens">Max New Tokens <span class="val" id="max-val">128</span></label>
                <input type="range" id="max_tokens" min="10" max="512" step="10" value="128">
            </div>

            <div class="form-group">
                <label for="top_k">Top-K Sampling <span class="val" id="topk-val">50</span></label>
                <input type="range" id="top_k" min="1" max="100" step="1" value="50">
            </div>

            <div class="form-group">
                <label for="top_p">Top-P (Nucleus) <span class="val" id="topp-val">0.90</span></label>
                <input type="range" id="top_p" min="0.10" max="1.00" step="0.05" value="0.90">
            </div>

            <div class="status-block">
                <div class="config-title" style="font-size: 0.90rem; border: none; padding-bottom: 0.25rem;">System Metrics</div>
                <div class="status-item">
                    <span class="label">Model:</span>
                    <span class="value" id="stat-model">Loading...</span>
                </div>
                <div class="status-item">
                    <span class="label">Total Params:</span>
                    <span class="value" id="stat-params">Loading...</span>
                </div>
                <div class="status-item">
                    <span class="label">Hardware:</span>
                    <span class="value" id="stat-device">Loading...</span>
                </div>
                <div class="status-item">
                    <span class="label">Inference Time:</span>
                    <span class="value" id="stat-time">-</span>
                </div>
            </div>
        </section>

        <!-- Generation workspace panel -->
        <section class="workspace-panel">
            <!-- input prompt -->
            <div class="input-area-card glass-card">
                <textarea id="prompt-input" placeholder="මෙතනින් සිංහල වාක්‍යයක් හෝ වචන කිහිපයක් ඇතුළත් කරන්න (උදා: ශ්‍රී ලංකාව යනු...)"></textarea>
                <div class="btn-row">
                    <button class="gen-btn" id="gen-button" onclick="generateText()">
                        <span class="spinner" id="gen-spinner"></span>
                        <span id="btn-text">Generate Response</span>
                    </button>
                </div>
            </div>

            <!-- generated output -->
            <div class="output-card glass-card">
                <div class="output-header">
                    <h2>Generated Outputs</h2>
                    <button class="copy-btn" id="copy-button" onclick="copyOutputText()" disabled>Copy Text</button>
                </div>
                <div class="output-display" id="output-box">
                    <span class="placeholder-text">ඇතුළත් කළ Prompt එක මත පදනම්ව ජනන වන සිංහල පෙළ මෙහි දර්ශනය වේ...</span>
                </div>
            </div>
        </section>
    </main>

    <script>
        // Update value displays for sliders dynamically
        const sliders = [
            { id: 'temperature', valId: 'temp-val', fixed: 2 },
            { id: 'max_tokens', valId: 'max-val', fixed: 0 },
            { id: 'top_k', valId: 'topk-val', fixed: 0 },
            { id: 'top_p', valId: 'topp-val', fixed: 2 }
        ];

        sliders.forEach(slider => {
            const input = document.getElementById(slider.id);
            const valueSpan = document.getElementById(slider.valId);
            input.addEventListener('input', () => {
                const parseVal = parseFloat(input.value);
                valueSpan.textContent = slider.fixed > 0 ? parseVal.toFixed(slider.fixed) : parseVal;
            });
        });

        // Load initialization status metadata from server
        async function loadSystemStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                document.getElementById('stat-model').textContent = data.model_path;
                document.getElementById('stat-params').textContent = (data.total_params / 1e6).toFixed(1) + 'M';
                document.getElementById('stat-device').textContent = data.device.toUpperCase();
            } catch (error) {
                console.error("Error loading system status: ", error);
            }
        }

        // Fire text generation API
        async function generateText() {
            const prompt = document.getElementById('prompt-input').value.trim();
            if (!prompt) {
                alert("පෙළ ජනනය කිරීමට පෙර කරුණාකර Prompt එකක් ඇතුළත් කරන්න.");
                return;
            }

            const genButton = document.getElementById('gen-button');
            const btnText = document.getElementById('btn-text');
            const spinner = document.getElementById('gen-spinner');
            const outputBox = document.getElementById('output-box');
            const copyBtn = document.getElementById('copy-button');

            // Set loading state
            genButton.disabled = true;
            spinner.style.display = 'inline-block';
            btnText.textContent = 'Generating...';
            outputBox.innerHTML = '';
            
            const cursorSpan = document.createElement('span');
            cursorSpan.className = 'cursor';
            outputBox.appendChild(cursorSpan);

            const payload = {
                prompt: prompt,
                max_new_tokens: parseInt(document.getElementById('max_tokens').value),
                temperature: parseFloat(document.getElementById('temperature').value),
                top_k: parseInt(document.getElementById('top_k').value),
                top_p: parseFloat(document.getElementById('top_p').value)
            };

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    // Update inference duration
                    document.getElementById('stat-time').textContent = result.time_ms + ' ms';
                    
                    // Display generated text with a clean, smooth typing typewriter display
                    typewriterEffect(outputBox, result.text, copyBtn);
                } else {
                    outputBox.innerHTML = `<span style="color: #ef4444;">Error generating response: ${result.message}</span>`;
                    genButton.disabled = false;
                    spinner.style.display = 'none';
                    btnText.textContent = 'Generate Response';
                }
            } catch (error) {
                outputBox.innerHTML = `<span style="color: #ef4444;">Server Connection Failed: Check running server.</span>`;
                genButton.disabled = false;
                spinner.style.display = 'none';
                btnText.textContent = 'Generate Response';
            }
        }

        // Simulate typing animation
        function typewriterEffect(container, text, copyButton) {
            container.innerHTML = '';
            
            const textNode = document.createElement('span');
            const cursorSpan = document.createElement('span');
            cursorSpan.className = 'cursor';
            container.appendChild(textNode);
            container.appendChild(cursorSpan);

            let i = 0;
            const speed_ms = text.length > 200 ? 5 : 15; // Type faster for longer generations
            
            function type() {
                if (i < text.length) {
                    textNode.textContent += text.charAt(i);
                    i++;
                    setTimeout(type, speed_ms);
                } else {
                    // End typing state, show final cleanup
                    cursorSpan.remove();
                    document.getElementById('gen-button').disabled = false;
                    document.getElementById('gen-spinner').style.display = 'none';
                    document.getElementById('btn-text').textContent = 'Generate Response';
                    copyButton.disabled = false;
                }
            }
            type();
        }

        // Copy button trigger
        function copyOutputText() {
            const text = document.getElementById('output-box').textContent;
            navigator.clipboard.writeText(text).then(() => {
                const copyBtn = document.getElementById('copy-button');
                copyBtn.textContent = 'Copied!';
                setTimeout(() => copyBtn.textContent = 'Copy Text', 2000);
            }).catch(err => {
                console.error('Could not copy output to clipboard: ', err);
            });
        }

        // Run status retrieval on page load
        window.addEventListener('DOMContentLoaded', loadSystemStatus);
    </script>
</body>
</html>
"""

# =============================================================================
# LIGHTWEIGHT WEB SERVER REQUEST HANDLER
# =============================================================================
class SinhalaGPTRequestHandler(http.server.BaseHTTPRequestHandler):

    # Disable default connection logging in terminal to keep model output logs clean
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        # 1. Server info status query
        if self.path == "/status":
            info = {
                "model_path": MODEL_PATH,
                "tokenizer_path": TOKENIZER_PATH,
                "device": device,
                "total_params": total_params,
                "model_load_time_s": model_load_time_s
            }
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(info).encode("utf-8"))
            return
            
        # 2. Main visual web page
        if self.path == "/" or self.path == "/index.html":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
            return
            
        # Fallback 404
        self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")

    def do_POST(self):
        if self.path == "/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)
            
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception as e:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Malformed JSON: {e}")
                return
                
            prompt = payload.get("prompt", "").strip()
            max_new_tokens = int(payload.get("max_new_tokens", 128))
            temperature = float(payload.get("temperature", 0.7))
            top_k = int(payload.get("top_k", 50))
            top_p = float(payload.get("top_p", 0.9))
            
            if not prompt:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "Prompt is empty"}).encode("utf-8"))
                return
                
            # Perform inference and record latency
            try:
                import torch
                
                print(f"\n[*] Generating for prompt: '{prompt}'")
                print(f"[Params] max_tokens={max_new_tokens}, temp={temperature}, top_k={top_k}, top_p={top_p}")
                
                t0 = time.time()
                
                # Tokenize input using SentencePiece
                # Unigram tokenizer model has pad_id = 0, unk_id = 1, bos_id = 2, eos_id = 3
                bos_id = tokenizer.bos_id()
                raw_ids = tokenizer.encode(prompt, out_type=int)
                input_ids = [bos_id] + raw_ids
                
                input_tensor = torch.tensor([input_ids]).to(device)
                
                # Run generation
                with torch.no_grad():
                    output_ids = model.generate(
                        input_tensor,
                        max_new_tokens=max_new_tokens,
                        do_sample=True if temperature > 0.0 else False,
                        temperature=temperature if temperature > 0.0 else None,
                        top_k=top_k if top_k > 0 else None,
                        top_p=top_p if top_p < 1.0 else None,
                        pad_token_id=0,
                        eos_token_id=3
                    )
                
                elapsed_ms = int((time.time() - t0) * 1000)
                
                # Extract generated completion IDs by chopping off the prompt tokens
                full_ids = output_ids[0].tolist()
                new_ids = full_ids[len(input_ids):]
                
                # Decode subword segments back to UTF-8 text
                completion_text = tokenizer.decode(new_ids)
                
                print(f"[+] Done. Generated {len(new_ids)} tokens in {elapsed_ms}ms ({round(len(new_ids)/(elapsed_ms/1000), 1)} tok/sec)")
                
                response_data = {
                    "status": "success",
                    "text": completion_text,
                    "time_ms": elapsed_ms
                }
                
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
                return
                
            except Exception as e:
                print(f"[-] Inference failed: {e}")
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
                return
                
        self.send_error(HTTPStatus.NOT_FOUND, "POST endpoint not found")

# =============================================================================
# MAIN RUNNER
# =============================================================================
def main():
    import socketserver
    
    # 1. Initialize weights and structures
    load_model_and_tokenizer()
    
    # 2. Setup server configuration
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), SinhalaGPTRequestHandler) as server:
            print(f"[*] Starting local server on: http://localhost:{PORT}")
            print(f"[*] Open http://localhost:{PORT} in your web browser to generate text.")
            print("[*] Press Ctrl+C to terminate the server.\n")
            server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n[!] Keyboard interrupt received. Shutting down web server. Goodbye!")
    except Exception as e:
        print(f"\n[!] Failed to start the server: {e}")

if __name__ == "__main__":
    main()

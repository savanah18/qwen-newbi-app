import * as vscode from 'vscode';

type ChatMessage = { id: number; sender: 'user' | 'assistant'; text: string };

// Triton Inference Server configuration
const TRITON_HTTP_URL = 'http://localhost:8000';
const TRITON_MODEL_NAME = 'qwen3-vl';
const TRITON_INFER_ENDPOINT = `${TRITON_HTTP_URL}/v2/models/${TRITON_MODEL_NAME}/infer`;
const TRITON_HEALTH_ENDPOINT = `${TRITON_HTTP_URL}/v2/health/ready`;
const TRITON_MODEL_READY_ENDPOINT = `${TRITON_HTTP_URL}/v2/models/${TRITON_MODEL_NAME}/ready`;
const MAX_MESSAGES_IN_UI = 100; // Limit chat display to prevent memory issues

export function activate(context: vscode.ExtensionContext): void {
  const startDisposable = vscode.commands.registerCommand('tritonAI.start', () => {
    vscode.window.showInformationMessage('🚀 Triton AI Chat Ready - Generic AI assistant powered by Triton Inference Server is active!');
  });

  const chatDisposable = vscode.commands.registerCommand('tritonAI.openChat', () => {
    ChatPanel.createOrShow(context);
  });

  context.subscriptions.push(startDisposable, chatDisposable);
}

export function deactivate(): void {
  ChatPanel.disposeInstance();
}

class ChatPanel {
  private static instance: ChatPanel | undefined;
  private readonly panel: vscode.WebviewPanel;
  private messages: ChatMessage[] = [
    {
      id: 1,
      sender: 'assistant',
      text: '🤖 Triton AI Chat Assistant\n\nHello! I\'m a versatile AI assistant powered by Triton Inference Server with Qwen3-VL-8B-Instruct. I support:\n• Real-time text generation (max 32 batch)\n• Embedding extraction and vectorization\n• Multimodal input (text + images)\n• Fast inference with response timing\n• Flexible model configuration\n\nWhat would you like to know or discuss?',
    },
  ];
  private nextId = 2;
  private serverStatus = 'Checking...';
  private modelLoaded = false;
  private isLoading = false;

  private constructor(private readonly context: vscode.ExtensionContext) {
    this.panel = vscode.window.createWebviewPanel(
      'tritonAIChat',
      'Triton AI Chat',
      vscode.ViewColumn.Beside,
      {
        enableScripts: true,
      },
    );

    this.panel.onDidDispose(() => this.dispose(), null, this.context.subscriptions);
    this.panel.webview.onDidReceiveMessage(msg => this.onMessage(msg));
    this.panel.webview.html = this.renderHtml();
    this.checkServerHealth();
  }

  static createOrShow(context: vscode.ExtensionContext): void {
    if (ChatPanel.instance) {
      ChatPanel.instance.panel.reveal(vscode.ViewColumn.Beside);
      ChatPanel.instance.pushState();
      return;
    }
    ChatPanel.instance = new ChatPanel(context);
  }

  static disposeInstance(): void {
    ChatPanel.instance?.dispose();
  }

  private dispose(): void {
    ChatPanel.instance = undefined;
  }

  private async checkServerHealth(): Promise<void> {
    try {
      // Check Triton server health
      const healthResponse = await fetch(TRITON_HEALTH_ENDPOINT);
      const modelResponse = await fetch(TRITON_MODEL_READY_ENDPOINT);
      
      if (healthResponse.ok && modelResponse.ok) {
        this.modelLoaded = true;
        this.serverStatus = `Triton: Ready | Model: ${TRITON_MODEL_NAME} loaded`;
      } else {
        this.serverStatus = 'Triton: Server running, model not ready';
        this.modelLoaded = false;
      }
    } catch {
      this.serverStatus = `Triton: Not running (${TRITON_HTTP_URL})`;
      this.modelLoaded = false;
    }
    this.pushState();
  }

  private async loadModel(): Promise<void> {
    // With Triton, models are loaded at startup - just check status
    await this.checkServerHealth();
    if (this.modelLoaded) {
      this.addMessage('assistant', `✓ Model ${TRITON_MODEL_NAME} is ready on Triton server`);
    } else {
      this.addMessage('assistant', 'Error: Model not loaded. Please start Triton server with the model.');
    }
  }

  private async onMessage(message: { type: string; text?: string; action?: string }): Promise<void> {
    if (message.type === 'send' && message.text) {
      const userMsg: ChatMessage = {
        id: this.nextId++,
        sender: 'user',
        text: message.text.trim(),
      };
      if (!userMsg.text) {
        return;
      }
      this.messages.push(userMsg);
      this.pushState(); // Show user message immediately

      if (!this.modelLoaded) {
        this.addMessage('assistant', 'Error: Triton server not ready. Please start the server first.');
        this.pushState();
        return;
      }

      // Add loading message
      this.isLoading = true;
      const loadingMsg: ChatMessage = {
        id: this.nextId++,
        sender: 'assistant',
        text: '⏳ Processing...',
      };
      this.messages.push(loadingMsg);
      this.pushState();

      try {
        const startTime = Date.now();
        console.log('[Triton AI] Sending request to Triton:', userMsg.text);
        
        // Build Triton inference request with mode parameter
        const tritonRequest = {
          inputs: [
            {
              name: 'message',
              shape: [1, 1],
              datatype: 'BYTES',
              data: [userMsg.text]
            },
            {
              name: 'mode',
              shape: [1, 1],
              datatype: 'BYTES',
              data: ['generate']  // Use 'generate' for text generation, 'embed' for embeddings
            }
          ],
          outputs: [
            { name: 'response' },
            { name: 'response_time' }
          ]
        };

        const response = await fetch(TRITON_INFER_ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(tritonRequest),
        });

        if (response.ok) {
          const data = (await response.json()) as { 
            outputs: Array<{ name: string; data: string[]; shape: number[] }> 
          };
          
          console.log('[Triton AI] Received response from Triton:', data);
          
          // Extract response and response_time from outputs
          const responseOutput = data.outputs.find(o => o.name === 'response');
          const timeOutput = data.outputs.find(o => o.name === 'response_time');
          
          // Handle both single and batch responses
          const responseText = responseOutput?.data[0] || 'No response';
          const responseTime = timeOutput?.data[0] || 0;
          const totalTime = (Date.now() - startTime) / 1000;
          
          console.log('[Triton AI] Total time:', totalTime, 'Model time:', responseTime);
          
          const displayText = `${responseText}\n\n⏱️ Model: ${Number(responseTime).toFixed(2)}s | Total: ${totalTime.toFixed(2)}s | Triton: ${TRITON_MODEL_NAME}`;
          
          // Remove loading message and add real response
          this.messages.pop();
          this.addMessage('assistant', displayText);
        } else {
          const errorText = await response.text();
          console.error('[Triton AI] Triton error:', errorText);
          this.messages.pop();
          this.addMessage('assistant', `Triton error: ${errorText}`);
        }
      } catch (error) {
        console.error('[Triton AI] Connection error:', error);
        this.messages.pop();
        this.addMessage('assistant', `Error: Cannot connect to Triton server - ${error}`);
      } finally {
        this.isLoading = false;
      }
      this.pushState();
    } else if (message.action === 'load-model') {
      await this.loadModel();
    } else if (message.action === 'clear-chat') {
      this.messages = [
        {
          id: this.nextId++,
          sender: 'assistant',
          text: 'Chat cleared.',
        },
      ];
      this.pushState();
    } else if (message.action === 'check-health') {
      await this.checkServerHealth();
    }
  }

  private addMessage(sender: 'user' | 'assistant', text: string): void {
    this.messages.push({
      id: this.nextId++,
      sender,
      text,
    });
  }

  private renderHtml(): string {
    const nonce = getNonce();
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Triton AI Chat</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; background: #0b1220; color: #e2e8f0; }
    header { padding: 12px 16px; background: #0f172a; border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
    .header-title { font-weight: 600; font-size: 14px; }
    .header-status { font-size: 12px; color: #94a3b8; }
    .controls { display: flex; gap: 6px; }
    #messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 8px; }
    .bubble { max-width: 85%; padding: 10px 12px; border-radius: 10px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }
    .assistant { background: #1e293b; align-self: flex-start; }
    .user { background: #1c3d5a; align-self: flex-end; }
    form { display: flex; flex-direction: column; gap: 8px; padding: 12px 16px; border-top: 1px solid #1f2937; background: #0f172a; }
    textarea { resize: none; padding: 10px; border-radius: 6px; border: 1px solid #334155; background: #111827; color: #e2e8f0; font-family: inherit; font-size: 14px; }
    textarea:focus { outline: none; border-color: #2563eb; }
    .button-group { display: flex; gap: 8px; }
    button { padding: 8px 12px; border: none; border-radius: 6px; background: #2563eb; color: #f8fafc; font-weight: 600; cursor: pointer; font-size: 13px; flex: 1; }
    button:hover { background: #1d4ed8; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    button.secondary { background: #334155; }
    button.secondary:hover { background: #475569; }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="header-title">🤖 Triton AI Chat</div>
      <div class="header-status" id="status-text">Checking Triton server...</div>
    </div>
    <div class="controls">
      <button onclick="checkHealth()" class="secondary" style="flex: 0 1 auto;">Check Status</button>
    </div>
  </header>
  <main id="messages" aria-live="polite"></main>
  <form id="chat-form">
    <textarea id="chat-input" placeholder="Type a message and press Send..." style="min-height: 60px; max-height: 120px;"></textarea>
    <div class="button-group">
      <button type="submit" id="send-btn">Send</button>
      <button type="button" onclick="clearChat()" class="secondary">Clear</button>
    </div>
  </form>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const messagesEl = document.getElementById('messages');
    const formEl = document.getElementById('chat-form');
    const inputEl = document.getElementById('chat-input');
    const statusEl = document.getElementById('status-text');
    const sendBtn = document.getElementById('send-btn');

    const render = (messages) => {
      messagesEl.innerHTML = '';
      messages.forEach(msg => {
        const div = document.createElement('div');
        div.className = \`bubble \${msg.sender}\`;
        div.textContent = msg.text;
        messagesEl.appendChild(div);
      });
      messagesEl.scrollTop = messagesEl.scrollHeight;
    };

    window.addEventListener('message', event => {
      const { type, messages, serverStatus, modelLoaded, isLoading } = event.data;
      if (type === 'state') {
        render(messages ?? []);
        if (serverStatus !== undefined) {
          statusEl.textContent = serverStatus;
        }
        if (modelLoaded !== undefined || isLoading !== undefined) {
          sendBtn.disabled = !modelLoaded || isLoading;
          inputEl.disabled = !modelLoaded || isLoading;
        }
      }
    });

    formEl.addEventListener('submit', event => {
      event.preventDefault();
      const text = inputEl.value.trim();
      if (!text) return;
      inputEl.value = '';
      vscode.postMessage({ type: 'send', text });
    });

    function loadModel() {
      vscode.postMessage({ action: 'load-model' });
    }

    function checkHealth() {
      vscode.postMessage({ action: 'check-health' });
    }

    function clearChat() {
      vscode.postMessage({ action: 'clear-chat' });
    }
  </script>
</body>
</html>`;
  }

  private pushState(): void {
    // Limit messages to prevent memory issues
    const messagesToSend = this.messages.slice(-MAX_MESSAGES_IN_UI);
    
    this.panel.webview.postMessage({
      type: 'state',
      messages: messagesToSend,
      serverStatus: this.serverStatus,
      modelLoaded: this.modelLoaded,
      isLoading: this.isLoading,
    });
  }
}

function getNonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < 16; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

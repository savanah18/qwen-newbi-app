import * as vscode from 'vscode';

type ChatMessage = { id: number; sender: 'user' | 'assistant'; text: string };

const MODEL_SERVER_URL = 'http://localhost:8000';
const CHAT_ENDPOINT = `${MODEL_SERVER_URL}/chat`;
const HEALTH_ENDPOINT = `${MODEL_SERVER_URL}/health`;
const LOAD_MODEL_ENDPOINT = `${MODEL_SERVER_URL}/load_model`;

export function activate(context: vscode.ExtensionContext): void {
  const helloDisposable = vscode.commands.registerCommand('dsaAgent.helloWorld', () => {
    vscode.window.showInformationMessage('🚀 DSA Agent Ready - Your data structure & algorithm learning companion is active!');
  });

  const chatDisposable = vscode.commands.registerCommand('dsaAgent.openChat', () => {
    ChatPanel.createOrShow(context);
  });

  context.subscriptions.push(helloDisposable, chatDisposable);
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
      text: '🤖 DSA Agent Activated\n\nHey! I\'m your agentic assistant for data structures, algorithms, and competitive programming. Ask me about:\n• Algorithm design and implementation\n• Data structure selection and optimization\n• Time/space complexity analysis\n• Interview problem strategies\n• Code optimization techniques\n\nLet\'s master DSA together!',
    },
  ];
  private nextId = 2;
  private serverStatus = 'Checking...';
  private modelLoaded = false;

  private constructor(private readonly context: vscode.ExtensionContext) {
    this.panel = vscode.window.createWebviewPanel(
      'dsaAgentChat',
      'DSA Agent Chat',
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
      const response = await fetch(HEALTH_ENDPOINT);
      if (response.ok) {
        const data = (await response.json()) as { model_loaded: boolean; status: string };
        this.modelLoaded = data.model_loaded;
        this.serverStatus = `Model: ${data.model_loaded ? 'Loaded' : 'Not loaded'} | Status: ${data.status}`;
      } else {
        this.serverStatus = 'Server error';
        this.modelLoaded = false;
      }
    } catch {
      this.serverStatus = 'Server not running (http://localhost:8000)';
      this.modelLoaded = false;
    }
    this.pushState();
  }

  private async loadModel(): Promise<void> {
    try {
      const response = await fetch(LOAD_MODEL_ENDPOINT, { method: 'POST' });
      if (response.ok) {
        const data = (await response.json()) as { message: string };
        this.addMessage('assistant', data.message);
        await this.checkServerHealth();
      } else {
        const data = (await response.json()) as { detail?: string };
        this.addMessage('assistant', `Error loading model: ${data.detail || 'Unknown error'}`);
      }
    } catch {
      this.addMessage('assistant', 'Error: Cannot connect to model server');
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

      if (!this.modelLoaded) {
        this.addMessage('assistant', 'Error: Model not loaded on server. Click "Load Model" first.');
        return;
      }

      try {
        const startTime = Date.now();
        const response = await fetch(CHAT_ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMsg.text, image_base64: null }),
        });

        if (response.ok) {
          const data = (await response.json()) as { response: string; response_time: number };
          const responseText = `${data.response}\n\n⏱️ Response time: ${data.response_time.toFixed(2)}s`;
          this.addMessage('assistant', responseText);
        } else {
          const data = (await response.json()) as { detail?: string };
          this.addMessage('assistant', `Server error: ${data.detail || 'Unknown error'}`);
        }
      } catch {
        this.addMessage('assistant', 'Error: Cannot connect to model server');
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
  <title>DSA Agent Chat</title>
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
      <div class="header-title">🤖 DSA Agent</div>
      <div class="header-status" id="status-text">Checking server...</div>
    </div>
    <div class="controls">
      <button onclick="loadModel()" class="secondary" style="flex: 0 1 auto;">Load Model</button>
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
      const { type, messages, serverStatus, modelLoaded } = event.data;
      if (type === 'state') {
        render(messages ?? []);
        if (serverStatus !== undefined) {
          statusEl.textContent = serverStatus;
        }
        if (modelLoaded !== undefined) {
          sendBtn.disabled = !modelLoaded;
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

    function clearChat() {
      vscode.postMessage({ action: 'clear-chat' });
    }
  </script>
</body>
</html>`;
  }

  private pushState(): void {
    this.panel.webview.postMessage({
      type: 'state',
      messages: this.messages,
      serverStatus: this.serverStatus,
      modelLoaded: this.modelLoaded,
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

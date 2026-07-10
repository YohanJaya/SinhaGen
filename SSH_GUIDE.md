# SSH Guide — GCP VM → RunPod GPU Training Workflow

This README covers everything you need to know about SSH, plus the specific steps for your setup: GCP VM (control center) → RunPod GPU pod (training) → GCS bucket (data/checkpoints).

---

## 1. What SSH Is

SSH (Secure Shell) is a protocol for securely connecting to a remote computer's command line over a network. Everything sent is encrypted. It's the standard way to manage remote servers, VMs, and cloud GPU instances.

Two components:
- **Client (`ssh`)** — what you run to connect *out* to another machine
- **Server (`sshd`)** — background service on the remote machine that *listens* for incoming connections

---

## 2. The `~/.ssh` Folder

`~/.ssh` is a hidden directory inside your home folder (`~` = home; the leading `.` hides it from a plain `ls`, so you need `ls -la` to see it). It's the standard location where SSH stores your identity and connection info.

**What typically lives inside it:**

| File | Purpose |
|------|---------|
| `id_ed25519` | Your **private key** — never share this, never leave your machine |
| `id_ed25519.pub` | Your **public key** — safe to share; paste this into RunPod/GitHub/etc |
| `authorized_keys` | On a machine you *receive* connections to — lists public keys allowed to log in as you |
| `known_hosts` | Cache of servers you've connected to before, so SSH can detect if a server's identity suddenly changes (protects against impersonation) |
| `config` | Your shortcut aliases (see Section 8) |

Every user account has its own `~/.ssh` folder. On your GCP VM it belongs to `yohan-jayasinghe`. If you SSH into RunPod later, that machine will have its own separate `~/.ssh` folder for whatever user you land as there (often `root`). If the folder doesn't exist yet, running `ssh-keygen` creates it automatically.

---

## 3. Basic Connection Syntax

```bash
ssh username@hostname
```

With extra options:

```bash
ssh -i ~/.ssh/id_ed25519 -p 2222 root@203.0.113.5
```

| Flag | Meaning |
|------|---------|
| `-i` | Path to the private key to authenticate with |
| `-p` | Custom port (default is 22; cloud providers often use a different one) |

---

## 4. Authentication Methods

**Password** — simple, but weaker. Many cloud providers (RunPod, GCP, AWS) disable it entirely for security.

**SSH key pair** — the standard for cloud work. Two matching files:

- **Private key** (`id_ed25519`) — stays on your machine, never shared, never uploaded anywhere
- **Public key** (`id_ed25519.pub`) — safe to share; you paste this into RunPod, GitHub, or a server's `~/.ssh/authorized_keys`

When you connect, the server challenges your client, and only the matching private key can respond correctly. That's why the `.pub` file is safe to share but the private key never is.

---

## 5. Generating a Key Pair

```bash
ssh-keygen -t ed25519 -C "some-label-for-yourself"
```

| Flag | Meaning |
|------|---------|
| `-t ed25519` | Key type — modern, fast, secure (preferred over old `rsa`) |
| `-C` | Just a comment/label, doesn't affect function |

Press **Enter** through all prompts to accept defaults and skip the passphrase (unless you want one). This saves:
- `~/.ssh/id_ed25519` (private)
- `~/.ssh/id_ed25519.pub` (public)

You can create multiple key pairs for different services — just give them different filenames when prompted (e.g., `~/.ssh/runpod_key`).

---

## 6. Checking Existing Keys

```bash
ls -la ~/.ssh
```

Lists everything in your SSH folder. Look for `id_ed25519` / `id_ed25519.pub` or similarly named files.

To view a public key's contents (safe to display/share):

```bash
cat ~/.ssh/id_ed25519.pub
```

---

## 7. Useful Everyday Commands

**Copy a file to a remote machine:**
```bash
scp localfile.txt username@host:/remote/path/
```

**Copy a file from remote to local:**
```bash
scp username@host:/remote/path/file.txt ./
```

**Copy an entire folder:**
```bash
scp -r local_folder/ username@host:/remote/path/
```

**Run a single command remotely without a full session:**
```bash
ssh username@host "ls -la /workspace"
```

---

## 8. SSH Config File (Shortcut Aliases)

Create/edit `~/.ssh/config`:

```
Host runpod
    HostName ssh.runpod.io
    User t7fwflfqvqlfq2-64410d45
    IdentityFile ~/.ssh/id_ed25519

Host gcpvm
    HostName 34.123.45.67
    User yohan-jayasinghe
    IdentityFile ~/.ssh/id_ed25519
```

Then just run:
```bash
ssh runpod
```
No need to retype the full command every time.

---

## 9. Keeping Sessions Alive: tmux

SSH sessions die if your connection drops (wifi blips, laptop sleeps) — this kills anything running inside, including model training. `tmux` creates a persistent terminal session on the remote machine that survives disconnects.

```bash
tmux new -s train        # start a named session
# ...run your training command here...
# Ctrl+B then D          # detach (session keeps running in background)

tmux attach -t train      # reattach later, from anywhere
tmux ls                    # list all active sessions
```

**Essential for any long-running training job.**

---

## 10. SSH Agent (Avoid Re-entering Passphrases)

If your key has a passphrase, `ssh-agent` remembers it for your session:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

---

## 11. Other Handy SSH Features

**Port forwarding** — access a remote service (Jupyter, TensorBoard) as if it's running locally:
```bash
ssh -L 8888:localhost:8888 username@host
```
Then open `localhost:8888` in your local browser.

**Efficient file sync** — `rsync` over SSH only transfers changes, faster than repeated `scp`:
```bash
rsync -avz -e ssh local_folder/ username@host:/remote/path/
```

---

## 12. Your Specific Workflow Checklist

- [ ] **Step 1:** On GCP VM, check for existing key: `ls -la ~/.ssh/id_ed25519.pub`
- [ ] **Step 2:** If missing, generate one: `ssh-keygen -t ed25519 -C "runpod-training"` (Enter through all prompts)
- [ ] **Step 3:** Display public key: `cat ~/.ssh/id_ed25519.pub`
- [ ] **Step 4:** Copy the full line (starts with `ssh-ed25519 AAAA...`)
- [ ] **Step 5:** Add it at https://www.runpod.io/console/user/settings → SSH Public Keys
- [ ] **Step 6:** Connect: `ssh t7fwflfqvqlfq2-64410d45@ssh.runpod.io -i ~/.ssh/id_ed25519`
- [ ] **Step 7:** Once connected, set up gcloud/gsutil auth on the RunPod pod to pull data from your GCS bucket
- [ ] **Step 8:** Pull tokenized data: `gsutil -m cp -r gs://your-bucket-name/tokenized-data /workspace/data/`
- [ ] **Step 9:** Get your decoder code onto the pod (git clone, or via the bucket)
- [ ] **Step 10:** Start training inside `tmux` so it survives disconnects
- [ ] **Step 11:** Periodically sync checkpoints back to GCS: `gsutil -m rsync -r /workspace/checkpoints gs://your-bucket-name/checkpoints`

---

## 13. Quick Troubleshooting

| Problem | Likely Cause |
|---------|--------------|
| `Permission denied (publickey)` | Public key not added to RunPod, or wrong `-i` path |
| `No such file or directory` on `~/.ssh/id_ed25519` | Key hasn't been generated yet |
| Connection just hangs | Wrong hostname/port, or firewall blocking |
| Training stopped after disconnect | Wasn't run inside `tmux` — always use `tmux` for long jobs |

# 🛡️ Security Policy

## 🩺 Our Commitment
We take the security of our biomedical imaging system and the privacy of vascular data seriously. If you discover a vulnerability, we appreciate your help in disclosing it to us in a responsible manner.

---

## ✅ Supported Versions
We currently provide security updates for the following versions:


| Version | Supported          |
| ------- | ------------------ |
| Main    | ✅ Yes             |
| < 1.0   | ❌ No              |

---

## 🚨 Reporting a Vulnerability
**Do not open a public issue for security vulnerabilities.** 🚫

If you find a security bug (such as a data leak, buffer overflow in C++ components, or a way to bypass the local Edge AI sandbox), please report it via:

1.  **GitHub Private Reporting:** Go to the Security Tab and select "Report a vulnerability."

### 📝 What to include in your report:
*   **Type of vulnerability:** (e.g., potential data exposure, local code execution).
*   **System Environment:** Your hardware setup (NIR sensor, Edge device model).
*   **Proof of Concept:** A brief description or script to reproduce the issue.

---

## 🕒 Our Process
*   **Response Time:** We will acknowledge your report within **48 hours**. ⏱️
*   **Updates:** We will provide a timeline for a fix and keep you updated on progress.
*   **Credit:** Once the fix is merged, we will give you credit for the discovery (unless you prefer to remain anonymous). 🤝

---

## 💡 Security Best Practices
As this system utilizes **Local Edge AI**, we recommend:
*   Running the inference engine on a dedicated, non-public network.
*   Ensuring that any raw NIR frames stored for 3D modeling are encrypted.
*   Keeping your local OS and Edge AI drivers (Jetson/Raspberry Pi) up to date.

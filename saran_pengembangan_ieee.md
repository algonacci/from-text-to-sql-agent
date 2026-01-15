# Roadmap Riset Jangka Panjang: The 8-Paper Series
Dokumen ini memetakan perjalanan riset granular dari fondasi hingga sistem otonom yang matang, aman, dan dapat dijelaskan.

---

# 🗺️ Phase I: The Core Trilogy (Sesuai Rencana Awal)

## 📘 Paper 1: The Foundation
**Tema**: *"Constructing a Foundational Text-to-SQL Pipeline"*
**Fokus**:
*   Membangun Pipeline Modular.
*   Komparasi Prompting Strategies (Zero/Few/CoT).
*   **Token Economics**: Analisis Trade-off Biaya vs Akurasi.
*   *Output*: Validated Pipeline Codebase.

## 📘 Paper 2: The Intelligence
**Tema**: *"Context-Awareness in Natural Language Querying"*
**Fokus**:
*   Menangani ambiguitas bahasa (Context retention).
*   Multi-turn conversation (Chat memory).
*   Peningkatan akurasi berbasis konteks historis.

## 📘 Paper 3: The Protocol
**Tema**: *"Standardizing Data Agents with Model Context Protocol (MCP)"*
**Fokus**:
*   Interoperabilitas sistem.
*   Standardisasi komunikasi antara Agent dan Database Server.
*   Implementasi MCP Server untuk SQL.

---

# 🗺️ Phase II: The Advanced Expansion (Paper 4-8)
*Topik-topik spesifik yang dipecah agar fokus pembahasan lebih tajam (High-impact Journals).*

## 📘 Paper 4: The Scalability
**Tema**: *"Efficient Schema Linking for Large-Scale Databases"*
**Masalah**: Context window penuh saat tabel ratusan.
**Fokus**:
*   **Schema Pruning**: Teknik vektor untuk memilih hanya tabel relevan.
*   Optimasi prompt untuk Enterprise ERP Schema.

## 📘 Paper 5: The Interaction
**Tema**: *"Human-in-the-Loop Mechanisms for Ambiguity Resolution"*
**Masalah**: AI menebak-nebak saat query ambigu.
**Fokus**:
*   Mekanisme Agent bertanya balik ke User.
*   Interactive Disambiguation UI.
*   Learning from user feedback.

## 📘 Paper 6: The Reliability
**Tema**: *"Self-Correction Cycles in Autonomous Query Generation"*
**Masalah**: SQL Error langsung gagal.
**Fokus**:
*   **Reflexion Loop**: Agent membaca error log DB dan memperbaiki SQL sendiri.
*   Execution-guided decoding.

## 📘 Paper 7: The Governance (Security)
**Tema**: *"A Multi-Layered Security Framework for Generative SQL"*
**Masalah**: SQL Injection & Data Leakage.
**Fokus**:
*   **Input Guardrails**: Mendeteksi malicious intents.
*   **AST Validation**: Memblokir statement destruktif (DROP/DELETE).
*   **Dynamic RBAC**: Pengaturan akses level row/column.

## 📘 Paper 8: The Trust (XAI)
**Tema**: *"Explainable Text-to-SQL: Tracing the Reasoning Path"*
**Masalah**: User tidak percaya hasil "Blackbox".
**Fokus**:
*   **Reasoning Trace**: Visualisasi alur logika ("Kenapa join tabel A?").
*   Natural Language Explanations of Query Logic.

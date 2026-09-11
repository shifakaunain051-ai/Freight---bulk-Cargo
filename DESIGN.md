# DESIGN.md — Nocturnal Intelligence Design System
## Reference: Google Stitch Project `12694042747017210949`

---

## 1. Brand Identity & Visual Language

- **Product Name**: Freight Intelligence Platform (Institutional Grade)
- **Aesthetic**: Modern Dark Glassmorphic, Mission-Control Analytical Dashboard
- **Personality**: Authoritative, restrained, information-dense, high-precision, executive SaaS
- **Target Audience**: Institutional bulk freight charterers, maritime logistics directors, commodity risk managers

---

## 2. Color Palette & Token System

### Core Palette (Dark Mode)
- **Base Background**: `#0B0E15` (Layer 0 canvas)
- **Card Surface (Layer 1)**: `#151A24`
- **Surface Hover / Elevated (Layer 2)**: `#1D2027` / `#272A32`
- **Surface Muted / Low**: `#10131A` / `#191B23`
- **Borders & Dividers**: `#262C3A` (Subtle container stroke), `#484555` (Active/focused stroke)

### Typography & Text Colors
- **On-Surface (Primary Text)**: `#E1E2EC` (High contrast, crisp legibility)
- **On-Surface-Variant (Muted Text)**: `#C9C4D8` / `#938EA1`
- **Accent Primary**: `#7C5CFC` (Electric Purple — primary actions, active tabs, glowing indicators)
- **Accent Secondary**: `#C9BFFF` (Soft Lavender — secondary data points, chart accents)
- **Accent Tertiary**: `#947DFF` (Interactive hover & focus rings)

### Semantic Colors
- **Success / Positive**: `#4ADE80` / `#3B5B43` (Rate drops for charterers, stable weather, low risk)
- **Warning / Moderate**: `#FBBF24` / `#B38E5D` / `#FFB77D` (Medium risk, monitor recommendations)
- **Danger / Alert**: `#F87171` / `#BA1A1A` / `#FFB4AB` (High cyclone risk, rate spikes, charter now alerts)

---

## 3. Typography Hierarchy

- **Font Family**: `Geist`, `JetBrains Mono`, `system-ui`, `sans-serif`
- **Tabular Figures**: `font-feature-settings: 'tnum' on, 'lnum' on` for all financial figures, dates, and KPI metrics.
- **Scale**:
  - `Display / KPI`: 36px – 48px, Weight 600–700, Line height 1.1, tracking -0.02em
  - `Headline LG`: 24px – 32px, Weight 600, Line height 1.25, tracking -0.01em
  - `Headline MD / Section`: 18px – 20px, Weight 600
  - `Body LG / MD`: 14px – 16px, Weight 400
  - `Label SM / Meta`: 11px – 12px, Weight 600, uppercase, tracking +0.05em
  - `Mono Data`: 12px – 13px, `JetBrains Mono`, Weight 400–500

---

## 4. Application Shell Layout

- **Fixed Sidebar Navigation (`w-64`, 256px)**:
  - Header: "FREIGHT INTELLIGENCE" + "INSTITUTIONAL GRADE" badge
  - Navigation links with Material Symbols Outlined icons and electric purple active pill indicators:
    1. Overview (`dashboard`)
    2. Forecast (`trending_up`)
    3. What-If Scenario (`tune`)
    4. Routes (`map`)
    5. Market (`show_chart`)
    6. Weather (`wb_sunny`)
    7. Data & Sources (`database`)
  - Footer: System health indicator pill (`● Live WAL Active`), Model v3 status.
- **Top Header Bar (`h-16`)**:
  - Screen title breadcrumb
  - Canonical Route quick-filter
  - Live data status & refresh trigger
- **Main Canvas (`ml-64 pt-16`)**:
  - 12-column grid layout, 16px–24px gutters, max-width 1440px
  - Glassmorphic elevated cards (`1px solid #262C3A`, radius `8px`–`12px`)

---

## 5. Canonical Trade Lanes (Strict Constraint)

All selection interfaces, comparisons, and forecast configurations are strictly bounded to the 5 production routes:

1. **Australia West Coast $\rightarrow$ East Coast India** | `Iron Ore` | `Capesize`
2. **Hay Point $\rightarrow$ East Coast India** | `Coal` | `Capesize`
3. **Hay Point $\rightarrow$ East Coast India** | `Coal` | `Panamax`
4. **Taboneo $\rightarrow$ East Coast India** | `Thermal Coal` | `Panamax`
5. **Taboneo $\rightarrow$ East Coast India** | `Thermal Coal` | `Supramax`

---

## 6. Backend API Integration Map

| Screen | Primary API Endpoint | Sub-Endpoints |
|---|---|---|
| **Overview Dashboard** | `GET /dashboard/overview` | `GET /data/status` |
| **Freight Forecast** | `POST /predict` | `GET /data/latest`, `GET /model/info` |
| **What-If Scenario** | `POST /predict/scenario` | `POST /predict` |
| **Route Intelligence** | `GET /analytics/routes` | `GET /analytics/freight-trends` |
| **Market Intelligence** | `GET /analytics/market-trends` | `GET /analytics/correlations`, `GET /analytics/freight-trends` |
| **Weather Intelligence** | `GET /analytics/weather-trends` | `GET /data/latest` |
| **Data & Sources** | `GET /model/info` | `GET /data/status`, `GET /data/telemetry` |

# NeuroLens — Brain Tumor Model Results Showcase

A modern, explainable MRI classification dashboard showcasing a TensorFlow-based brain tumor detection model. Built with React, tRPC, and a dark-mode-first "Clinical Signal" design system.

## Overview

NeuroLens is an interactive research tool that allows medical professionals and researchers to:

- Upload MRI brain scans for tumor classification
- View real-time model predictions with confidence scores
- Visualize model attention maps (Grad-CAM overlays)
- Process batch uploads and generate PDF reports
- Understand model behavior through explainability features

**Key Distinction:** This application honestly presents model outputs without fabricating accuracy claims. The included model shows classification limitations and includes confidence levels for transparency.

## Features

✅ **Real-time Inference** - Submit MRI images and receive instant predictions  
✅ **Confidence Visualization** - Full probability distribution across 4 tumor classes  
✅ **Attention Maps** - Grad-CAM overlays show where the model focused  
✅ **Batch Processing** - Analyze multiple images in one session  
✅ **PDF Export** - Generate downloadable reports with images and predictions  
✅ **Dark/Light Themes** - Optimized for clinical environments  
✅ **Error Handling** - Per-image validation and clear error messaging  
✅ **Responsive Design** - Works on desktop, tablet, and mobile

## Architecture

### Tech Stack

**Frontend:**

- React 19 + TypeScript
- Vite (build tool)
- Tailwind CSS + shadcn/ui components
- tRPC client for type-safe API calls
- Wouter for routing

**Backend:**

- Express.js server
- tRPC for API routes
- TensorFlow.js runtime for model inference
- Drizzle ORM for database
- MySQL for persistence

**ML Model:**

- TensorFlow Keras (brain_tumor_model.keras)
- EfficientNet backbone
- Input: 224×224×3 RGB images
- Output: 4-class softmax (glioma, meningioma, notumor, pituitary)

### Project Structure

```
brain-tumor-results-showcase/
├── client/                 # React frontend
│   ├── src/
│   │   ├── App.tsx        # Main app component
│   │   ├── pages/         # Page components (Home, NotFound)
│   │   ├── components/    # Reusable React components
│   │   ├── contexts/      # React context providers (ThemeContext)
│   │   ├── hooks/         # Custom React hooks
│   │   ├── lib/           # tRPC client, utilities
│   │   └── index.css      # Tailwind + theme variables
│   ├── index.html         # Entry point
│   └── vite.config.ts     # Vite configuration
│
├── server/                 # Express + tRPC backend
│   ├── index.ts           # Server entry point
│   ├── routers.ts         # tRPC route definitions
│   ├── inference.ts       # Model inference logic
│   ├── _core/             # Core server utilities
│   │   ├── trpc.ts        # tRPC setup
│   │   ├── context.ts     # tRPC context
│   │   ├── llm.ts         # AI integrations
│   │   ├── env.ts         # Environment config
│   │   └── ...
│   └── *.test.ts          # Backend tests
│
├── shared/                 # Shared types & constants
│   ├── types.ts           # TypeScript interfaces
│   ├── const.ts           # Constants
│   └── model.ts           # Model-related types
│
├── drizzle/               # Database schema & migrations
│   ├── schema.ts          # Drizzle ORM schema
│   ├── relations.ts       # Table relationships
│   └── migrations/        # Migration files
│
├── model/
│   └── brain_tumor_model.keras  # TensorFlow model artifact
│
├── patches/               # pnpm dependency patches
│
├── package.json           # Dependencies & scripts
├── vite.config.ts         # Vite build config
├── vitest.config.ts       # Vitest test config
├── tsconfig.json          # TypeScript config
├── drizzle.config.ts      # Drizzle ORM config
├── Dockerfile             # Container image
├── audit_findings.md      # Model audit results & technical notes
├── ideas.md               # Design system & brand guidelines
└── README.md              # This file
```

## Getting Started

### Prerequisites

- Node.js 18+ or pnpm 10+
- Python 3.8+ (for optional ML scripts)
- MySQL database (optional; embedded for development)

### Installation

1. **Clone and install:**

```bash
cd brain-tumor-results-showcase
pnpm install
```

2. **Set up environment variables:**

```bash
# Create .env.local or .env file
cp .env.example .env.local  # If example exists

# Add required variables (see Environment Variables section)
```

3. **Push database schema (if using MySQL):**

```bash
pnpm run db:push
```

### Running Locally

**Development mode** (with hot reload):

```bash
pnpm run dev
```

Server runs on `http://localhost:3000`

**Production build:**

```bash
pnpm run build
pnpm run start
```

**Type checking:**

```bash
pnpm run check
```

**Run tests:**

```bash
pnpm run test
```

**Format code:**

```bash
pnpm run format
```

## Environment Variables

Create a `.env.local` file with:

```env
# Server
NODE_ENV=development
PORT=3000

# Database (optional)
DATABASE_URL=mysql://user:password@host:3306/database_name

# Authentication & OAuth (optional)
JWT_SECRET=your-secret-key
OAUTH_SERVER_URL=https://your-oauth-provider

# Analytics (optional)
VITE_ANALYTICS_ENDPOINT=https://your-analytics.com
VITE_ANALYTICS_WEBSITE_ID=your-website-id

# App metadata
VITE_APP_TITLE=NeuroLens — Brain Tumor Model Results
VITE_APP_ID=your-app-id
```

## Model Details

### Classification Task

4-class brain tumor classification from MRI scans:

1. **Glioma** - Most common malignant tumor
2. **Meningioma** - Tumor from dura mater
3. **No Tumor** - Normal brain scan
4. **Pituitary** - Pituitary gland tumor

### Model Architecture

- **Backbone:** EfficientNet
- **Input:** 224×224×3 RGB images
- **Preprocessing:** Raw 0–255 pixel values (model includes rescaling layer)
- **Output:** 4-unit softmax (probability distribution)

### Important Limitations

⚠️ **Model Audit Findings:**

The included model shows significant classification issues:

- Misclassifies many tumor cases as "no tumor" (80%+ confidence)
- Test set performance is poor across multiple labeled examples
- This is a **model artifact/training quality problem**, not a preprocessing issue

**Examples from test set:**
| File | User Label | Model Prediction | Confidence |
|---|---|---|---|
| testg.jpeg | glioma | notumor | 81.33% |
| test2-m.jpg | meningioma | notumor | 65.77% |
| test3-n.jpeg | no tumor | notumor | 90.61% ✓ |
| test4-p.jpg | pituitary | notumor | 80.39% |

**Fixing requires:**

- Original labeled training/validation dataset
- Model retraining with proper evaluation metrics
- Cross-validation on held-out test set
- Documented preprocessing & class mapping

The app correctly displays this behavior—it does NOT fabricate confidence or relabel outputs.

## API Routes

All API endpoints are served via tRPC at `/api/trpc/`.

### Key Procedures

**Authentication:**

- `auth.me` — Get current user
- `auth.logout` — Logout and clear session

**Image Analysis:**

- `analyze.mutation` — Submit image for inference
  - Input: `{ imageDataUrl: string, mimeType?: string }`
  - Output: `{ predictions: {...}, gradcam: [...], error?: string }`

## Design System: "Clinical Signal"

The interface follows Swiss International typographic principles adapted for medical research:

### Color Palette

- **Background:** Warm ivory (`#f9f7f4`)
- **Text:** Deep ink (`#1a1410`)
- **Accent (Primary):** Sea-glass teal (`#2d9b94`) — Model activity & explainability
- **Accent (Warning):** Restrained coral (`#e07856`) — Uncertainty & attention
- **Dark mode:** Inverted palette for reduced eye strain

### Typography

- **UI Labels:** DM Sans (sans-serif, precise)
- **Headlines:** IBM Plex Serif (serif, editorial weight)
- **Numbers:** Tabular figures for data alignment

### Motion

- **Transitions:** 160–240ms ease-out
- **Stagger:** 40ms between result plates
- **Respect:** `prefers-reduced-motion` media query

### Component Philosophy

- **Evidence before decoration** — Visual hierarchy serves clarity
- **Asymmetric composition** — Left rail context, main canvas results
- **Generous whitespace** — Calming, legible scanning
- **Interactive revelation** — Hover/select exposes model reasoning

## Development Workflow

### Project Management

- Use `audit_findings.md` for model limitations & technical decisions
- Consult `ideas.md` for design rationale and brand voice

### Database

- Schema defined in [drizzle/schema.ts](drizzle/schema.ts)
- Run migrations: `pnpm run db:push`
- View migrations: `drizzle/migrations/`

### Frontend Components

- Component library in [client/src/components/](client/src/components/)
- UI primitives in [client/src/components/ui/](client/src/components/ui/) (shadcn/ui)
- Add new pages to [client/src/pages/](client/src/pages/)

### Backend Routes

- tRPC procedures in [server/routers.ts](server/routers.ts)
- Add new routers and nest them under `appRouter`
- Server context in [server/\_core/context.ts](server/_core/context.ts)

### Adding New Features

1. Create tRPC procedure in `server/routers.ts`
2. Call via `client/lib/trpc.ts`
3. Handle in React component with hooks
4. Add tests alongside feature code

## Testing

```bash
# Run all tests
pnpm run test

# Test specific file
pnpm run test server/inference.test.ts

# Watch mode
pnpm run test -- --watch
```

## Docker Deployment

Build and run the Docker image:

```bash
docker build -t neurolens .
docker run -p 3000:3000 -e NODE_ENV=production neurolens
```

## Troubleshooting

**Port already in use:**

```bash
lsof -i :3000
kill -9 <PID>
```

**Database connection error:**

- Verify `DATABASE_URL` env var is set
- Check MySQL server is running
- Run `pnpm run db:push`

**Model inference fails:**

- Check model file exists: `model/brain_tumor_model.keras`
- Verify image is valid RGB (not grayscale or corrupted)
- Check server logs for detailed error

**OAuth not configured:**

- This is optional; app runs without `OAUTH_SERVER_URL`
- Set only if implementing authentication

## Contributing

1. Create feature branches from `main`
2. Run `pnpm run check` and `pnpm run test` before committing
3. Use `pnpm run format` for consistent code style
4. Write clear commit messages
5. Submit pull requests with descriptions

## Security Considerations

⚠️ **Important:**

- Never commit `.env.local` or `.env` files
- Remove `.project-config.json` (contains secrets)
- Use environment variables for all credentials
- Run `pnpm audit` regularly for dependency vulnerabilities
- Model outputs are **not medical advice**—this is a research tool

## License

MIT

## Contact

For questions or issues, please reach out or file an issue in the repository.

---

**Last Updated:** August 2026  
**Model Status:** Research artifact (accuracy limitations noted in audit_findings.md)  
**Build:** NeuroLens v1.0.0

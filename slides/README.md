# Slides - From Text to SQL Agent

Presentation slides untuk tech talk "From Text to SQL Agent - Smart Query in Action"

## Setup

### Requirements
- Quarto
- PlantUML (untuk render diagram)

### Generate Slides

```bash
quarto render main.qmd
```

### Preview Slides

```bash
quarto preview main.qmd
```

## Render PlantUML Diagram

Untuk render arsitektur sistem diagram:

**Option 1: Using PlantUML CLI**
```bash
plantuml arsitektur-sistem.puml -o diagrams/
```

**Option 2: Using Online Editor**
1. Buka https://www.plantuml.com/plantuml/uml/
2. Copy paste content dari `arsitektur-sistem.puml`
3. Download sebagai PNG/JPG
4. Save ke folder `diagrams/arsitektur-sistem.jpg`

**Option 3: Using VS Code Extension**
1. Install extension: PlantUML
2. Open `arsitektur-sistem.puml`
3. Press `Alt+D` untuk preview
4. Export sebagai PNG/JPG ke folder `diagrams/`

## Folder Structure

```
slides/
├── main.qmd                    # Main presentation file
├── styles.css                  # Custom CSS styles
├── profile.jpg                 # Profile photo
├── logo.png                    # Logo (if any)
├── arsitektur-sistem.puml      # PlantUML diagram source
├── diagrams/
│   └── arsitektur-sistem.jpg   # Rendered diagram (placeholder)
└── memes/
    ├── hello-hello-everybody.jpg
    ├── nice-to-meet-you.jpg
    ├── tau-apa-tentang-sql-agent.png
    └── placeholder-demo.jpg    # Demo screenshot placeholder
```

## Notes

- Semua emoji sudah dihapus dari slides
- PlantUML diagram di-render sebagai static image untuk performa lebih baik
- Meme images perlu disiapkan di folder `memes/`
- Diagram perlu di-render ke folder `diagrams/`

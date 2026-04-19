# Recomendador de Evaluadores de Tesinas - FCEFyN UNC

Aplicación web para sugerir evaluadores (miembros de tribunal) para tesinas nuevas, 
basándose en el historial de tesinas pasadas y la similitud temática de los títulos.

**URL de producción**: https://evaluadorestesinas-fcefyn-aguluna.streamlit.app

## Archivos

- `app.py` — Código de la aplicación Streamlit
- `tesinas.xlsx` — Base de datos (hoja única "Tesinas" con 10 columnas)
- `fusiones.txt` — Desambiguación de variantes ortográficas de nombres
- `requirements.txt` — Dependencias de Python

## Cómo actualizar con una tesina nueva

1. Descargar `tesinas.xlsx` desde este repositorio
2. Abrirlo en Excel / LibreOffice / Google Sheets
3. Agregar una nueva fila en la hoja `Tesinas` con las 10 columnas:
   - Año
   - Tesinista
   - Título del plan
   - Director
   - Codirector
   - Evaluadores propuestos (separar con punto y coma)
   - Evaluadores recusados (separar con punto y coma)
   - Tribunal 1 (titular)
   - Tribunal 2
   - Tribunal 3
4. Guardar el archivo
5. Subirlo a este repositorio reemplazando la versión anterior
6. La app se actualiza automáticamente en ~1 minuto

## Cómo corregir un typo en un nombre

Si detectás que en una tesina nueva aparece un nombre con typo (ej. "Jaklin Kembro" 
cuando la persona correcta es "Jackelyn Kembro"), agregá una línea al final de 
`fusiones.txt` con el formato:

```
Jaklin Kembro => Jackelyn Kembro
```

## Técnica

- **Framework**: Streamlit
- **Matching semántico**: `sentence-transformers` con modelo multilingüe 
  `paraphrase-multilingual-MiniLM-L12-v2`
- **Ranking**: combina similitud semántica (70%), penalización por carga reciente 
  (20%) y bonus por actividad reciente (10%)

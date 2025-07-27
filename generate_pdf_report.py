#!/usr/bin/env python3
"""
Generate PDF Report for Composed Image Retrieval System
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os

def create_pdf_report():
    """Create a comprehensive PDF report with flowchart and documentation"""
    
    # Create PDF document
    doc = SimpleDocTemplate("Composed_Image_Retrieval_System_Report.pdf", 
                           pagesize=A4,
                           rightMargin=72,
                           leftMargin=72,
                           topMargin=72,
                           bottomMargin=72)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12,
        textColor=colors.darkgreen
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=10,
        spaceAfter=6,
        fontName='Courier',
        leftIndent=20
    )
    
    # Build the story (content)
    story = []
    
    # Title Page
    story.append(Paragraph("Composed Image Retrieval System", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Technical Documentation & Flowchart", heading_style))
    story.append(Spacer(1, 30))
    story.append(Paragraph("A comprehensive guide to the fashion image retrieval system with color filtering capabilities", normal_style))
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", heading_style))
    story.append(Spacer(1, 12))
    
    toc_items = [
        ["1.", "System Overview", "3"],
        ["2.", "Architecture Components", "4"],
        ["3.", "Retrieval Process Flowchart", "5"],
        ["4.", "Detailed Process Steps", "6"],
        ["5.", "Color Processing System", "7"],
        ["6.", "Performance Metrics", "8"],
        ["7.", "API Endpoints", "9"],
        ["8.", "Technical Specifications", "10"]
    ]
    
    toc_table = Table(toc_items, colWidths=[0.5*inch, 4*inch, 0.5*inch])
    toc_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(toc_table)
    story.append(PageBreak())
    
    # 1. System Overview
    story.append(Paragraph("1. System Overview", heading_style))
    story.append(Paragraph("""
    The Composed Image Retrieval (CIR) system is an advanced fashion recommendation platform that combines 
    visual and textual information to find similar fashion items. Users upload a reference image and provide 
    modification text (e.g., "make it greenish-yellow"), and the system returns the most similar items 
    that match both the visual style and the requested modifications.
    """, normal_style))
    
    story.append(Paragraph("Key Features:", subheading_style))
    story.append(Paragraph("• Multi-modal search combining image and text", normal_style))
    story.append(Paragraph("• Advanced color filtering with 80+ color combinations", normal_style))
    story.append(Paragraph("• Category-specific optimization (Dress, Shirt, TopTee)", normal_style))
    story.append(Paragraph("• Real-time search with 100-500ms response time", normal_style))
    story.append(Paragraph("• Support for 37,189 fashion images", normal_style))
    story.append(PageBreak())
    
    # 2. Architecture Components
    story.append(Paragraph("2. Architecture Components", heading_style))
    
    components_data = [
        ["Component", "Description", "Specifications"],
        ["CLIP Model", "Vision-language feature extraction", "RN50x4 variant, 640-dim image features, 512-dim text features"],
        ["Combiner Network", "Feature fusion with attention", "AttentionFusionCombiner, 640-dim joint features"],
        ["HNSW Index", "Fast similarity search", "Hierarchical Navigable Small World, category-specific indices"],
        ["Color Clustering", "Color-based filtering", "K-means clustering + KD-trees, adaptive radius"],
        ["Flask API", "RESTful service", "CORS enabled, JSON responses, image serving"],
        ["Frontend", "User interface", "HTML/JavaScript, drag-and-drop upload, real-time results"]
    ]
    
    comp_table = Table(components_data, colWidths=[1.5*inch, 2.5*inch, 2*inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
    ]))
    story.append(comp_table)
    story.append(PageBreak())
    
    # 3. Retrieval Process Flowchart
    story.append(Paragraph("3. Retrieval Process Flowchart", heading_style))
    story.append(Paragraph("""
    The following flowchart illustrates the complete retrieval process from user input to result display:
    """, normal_style))
    
    # ASCII Flowchart in PDF
    flowchart_text = """
START
  |
  v
[User Uploads Image + Text]
  |
  v
[Frontend Sends API Request]
  |
  v
[API Service Receives Request]
  |
  v
[Decode Base64 Image] -----> [Extract Text]
  |                           |
  v                           v
[CLIP Image Encoding]     [CLIP Text Encoding]
  |                           |
  v                           v
[640-dim Image Features]  [512-dim Text Features]
  |                           |
  +---------------------------+
  |
  v
[Combiner Network - Feature Fusion]
  |
  v
[640-dim Joint Features]
  |
  v
[Color Analysis from Text]
  |
  v
{Color Detected?}
  |
  YES -----> [Color Pre-filtering] -----> [Get Color-Filtered Images]
  |                                         |
  NO                                        |
  |                                         |
  v                                         v
[Category Selection] <----------------------+
  |
  v
{Multiple Categories?}
  |
  YES -----> [Build Combined Index]
  |           |
  NO          |
  |           |
  v           v
[Use Category Index] <------+
  |
  v
[HNSW Similarity Search]
  |
  v
[Get Top 100 Candidates]
  |
  v
{Color Filter Active?}
  |
  YES -----> [Apply Color Filter] -----> [Select Top 20 Color Matches]
  |                                       |
  NO                                      |
  |                                       |
  v                                       v
[Select Top 20 Overall] <-----------------+
  |
  v
[Prepare Result Objects]
  |
  v
[Add Image Paths & Metadata]
  |
  v
[Calculate Similarity Scores]
  |
  v
[Send JSON Response to Frontend]
  |
  v
[Display Images to User]
  |
  v
END
    """
    
    story.append(Paragraph(flowchart_text, code_style))
    story.append(PageBreak())
    
    # 4. Detailed Process Steps
    story.append(Paragraph("4. Detailed Process Steps", heading_style))
    
    steps_data = [
        ["Step", "Process", "Output"],
        ["1", "Image Decoding", "PIL Image object from base64"],
        ["2", "CLIP Preprocessing", "Normalized tensor (224x224)"],
        ["3", "Image Feature Extraction", "640-dimensional vector"],
        ["4", "Text Tokenization", "CLIP tokenized text"],
        ["5", "Text Feature Extraction", "512-dimensional vector"],
        ["6", "Feature Fusion", "640-dimensional joint features"],
        ["7", "Color Analysis", "Target color + radius"],
        ["8", "Category Selection", "Search categories list"],
        ["9", "Index Selection", "HNSW index for search"],
        ["10", "Similarity Search", "Top 100 candidate indices"],
        ["11", "Color Filtering", "Color-filtered image IDs"],
        ["12", "Result Preparation", "Top 20 formatted results"],
        ["13", "Response Generation", "JSON with metadata"]
    ]
    
    steps_table = Table(steps_data, colWidths=[0.5*inch, 2.5*inch, 2*inch])
    steps_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
    ]))
    story.append(steps_table)
    story.append(PageBreak())
    
    # 5. Color Processing System
    story.append(Paragraph("5. Color Processing System", heading_style))
    story.append(Paragraph("""
    The color processing system analyzes modification text to extract color information and applies 
    intelligent filtering to find images with matching colors.
    """, normal_style))
    
    story.append(Paragraph("Color Detection Methods:", subheading_style))
    story.append(Paragraph("1. Exact Color Matching: blue, red, green, etc.", normal_style))
    story.append(Paragraph("2. Color Combinations: green-yellow, blue-green, etc.", normal_style))
    story.append(Paragraph("3. Color Modifiers: light-blue, dark-red, etc.", normal_style))
    story.append(Paragraph("4. Fashion Colors: turquoise, lavender, coral, etc.", normal_style))
    
    story.append(Paragraph("Adaptive Radius System:", subheading_style))
    radius_data = [
        ["Query Type", "Radius", "Description"],
        ["Combination", "100", "For 'ish', 'combination', 'mix' words"],
        ["Light/Dark", "80", "For 'light', 'dark', 'pale' modifiers"],
        ["Exact", "60", "For precise color matches"]
    ]
    
    radius_table = Table(radius_data, colWidths=[2*inch, 1*inch, 3*inch])
    radius_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(radius_table)
    story.append(PageBreak())
    
    # 6. Performance Metrics
    story.append(Paragraph("6. Performance Metrics", heading_style))
    
    metrics_data = [
        ["Metric", "Value", "Description"],
        ["Total Images", "37,189", "Unique fashion items across all categories"],
        ["Dress Images", "11,643", "Dress category items"],
        ["Shirt Images", "13,261", "Shirt category items"],
        ["TopTee Images", "12,945", "TopTee category items"],
        ["Search Speed", "100-500ms", "Average response time per query"],
        ["Color Support", "80+", "Supported colors and combinations"],
        ["Result Count", "20", "Top results returned per query"],
        ["Feature Dimensions", "640", "Joint feature vector size"],
        ["Index Type", "HNSW", "Hierarchical Navigable Small World"],
        ["Color Clusters", "K-means", "K-means clustering for color filtering"]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
    ]))
    story.append(metrics_table)
    story.append(PageBreak())
    
    # 7. API Endpoints
    story.append(Paragraph("7. API Endpoints", heading_style))
    
    endpoints_data = [
        ["Endpoint", "Method", "Description", "Response"],
        ["/health", "GET", "Health check", "Status: healthy"],
        ["/categories", "GET", "Get available categories", "List of categories"],
        ["/retrieve", "POST", "Main retrieval endpoint", "JSON with results"],
        ["/images/<id>", "GET", "Serve image files", "Image file"]
    ]
    
    endpoints_table = Table(endpoints_data, colWidths=[1.5*inch, 1*inch, 2.5*inch, 1*inch])
    endpoints_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    story.append(endpoints_table)
    
    story.append(Paragraph("Sample /retrieve Request:", subheading_style))
    sample_request = """
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "modification_text": "make it greenish-yellow",
  "search_categories": ["dress", "shirt", "toptee"]
}
    """
    story.append(Paragraph(sample_request, code_style))
    story.append(PageBreak())
    
    # 8. Technical Specifications
    story.append(Paragraph("8. Technical Specifications", heading_style))
    
    tech_specs_data = [
        ["Component", "Technology", "Version/Specs"],
        ["Backend Framework", "Flask", "Python web framework"],
        ["Deep Learning", "PyTorch", "Neural network framework"],
        ["Vision Model", "CLIP", "RN50x4 variant"],
        ["Search Index", "HNSW", "Hierarchical Navigable Small World"],
        ["Color Clustering", "scikit-learn", "K-means + KD-trees"],
        ["Image Processing", "Pillow", "PIL fork for image handling"],
        ["Vector Operations", "NumPy", "Numerical computing"],
        ["Frontend", "HTML/JavaScript", "Vanilla JS with drag-drop"],
        ["API Format", "JSON", "RESTful API with CORS"],
        ["Image Serving", "Flask static", "Direct file serving"],
        ["Environment", "Python venv", "Isolated dependencies"]
    ]
    
    tech_table = Table(tech_specs_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
    ]))
    story.append(tech_table)
    
    story.append(Spacer(1, 20))
    story.append(Paragraph("Conclusion:", subheading_style))
    story.append(Paragraph("""
    The Composed Image Retrieval System represents a sophisticated approach to fashion recommendation, 
    combining state-of-the-art vision-language models with intelligent color processing and optimized 
    search algorithms. The system provides fast, accurate, and user-friendly fashion discovery capabilities 
    with support for complex color combinations and multi-category search.
    """, normal_style))
    
    # Build the PDF
    doc.build(story)
    print("✅ PDF report generated successfully: Composed_Image_Retrieval_System_Report.pdf")

if __name__ == "__main__":
    create_pdf_report() 
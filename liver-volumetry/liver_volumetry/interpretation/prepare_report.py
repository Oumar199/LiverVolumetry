"""Prepare a medical report for the liver volumetry project
"""

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def generate_medical_report(output_pdf: str, patient_name: str, seg_path: str, clinical_analysis: str, illustration_path:str = "images/20260217_2225_Liver Volumetry Analysis_remix_01khpv8yqrekbvrwrh52jb4m22.png"):
    
    # Marges ajustées pour garantir le format une page
    doc = SimpleDocTemplate(output_pdf, pagesize=A4, 
                            topMargin=15, bottomMargin=15, leftMargin=30, rightMargin=30)
    story = []
    styles = getSampleStyleSheet()

    # --- 1. Illustration
    banner = Image(illustration_path, width=7.5*inch, height=2.8*inch)
    story.append(banner)
    
    # Line of separation
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.dodgerblue, spaceAfter=10))

    # --- 2. Title and Patient Info ---
    title_style = ParagraphStyle('T', parent=styles['Title'], textColor=colors.dodgerblue, fontSize=22, alignment=0)
    story.append(Paragraph("LIVER ANALYSIS REPORT", title_style))
    
    patient_text = f"<b>Patient:</b> {patient_name} <br/> <b>Date:</b> February 19, 2026"
    story.append(Paragraph(patient_text, styles['Normal']))
    story.append(Spacer(1, 10))

    # --- 3. Segmentation ---
    story.append(Paragraph("<b>SEGMENTATION VIEW</b>", ParagraphStyle('H', parent=styles['Normal'], textColor=colors.darkblue)))
    # width 7.4 / Height 3.2
    seg_img = Image(seg_path, width=7.4*inch, height=3.2*inch)
    story.append(seg_img)
    
    # Separation line
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceBefore=10, spaceAfter=10))

    # --- 4. Clinical analysis ---
    analysis_style = ParagraphStyle('Analysis', parent=styles['Normal'], fontSize=11, leading=19) 
    
    story.append(Paragraph(f"""<b>Clinical Assessment & Findings:</b><br/>
{clinical_analysis.replace('**', "<strong>")}""", analysis_style))

    # Génération du PDF
    doc.build(story)
    print(f"Report successfully generated: {output_pdf}")

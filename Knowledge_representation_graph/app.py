import os
from flask import Flask, render_template, request
import spacy
from newspaper import Article
from werkzeug.utils import secure_filename
from pyvis.network import Network
import fitz

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

def get_color_for_entity(node_text, entity_map):
    color_map = {
        "ORG": "#FFA07A",    # Light Salmon
        "PERSON": "#87CEFA", # Light Sky Blue
        "GPE": "#98FB98",    # Pale Green
        "NORP": "#FFD700",   # Gold
        "PRODUCT": "#FFB6C1", # Light Pink
        "DISEASE": "#E6E6FA"  # Lavender
    }
    for entity_text, label in entity_map.items():
        if node_text.strip().lower() == entity_text.strip().lower():
            return color_map.get(label, "#D3D3D3") # Light Gray default
    return "#D3D3D3"


def fix_verb_label(verb):
    corrections = {
        "develop": "developed",
        "coordinate": "coordinated",
        "focus": "focusing on",
        "play": "developed",
        "lead": "led to",
        "cause": "caused by"
    }
    return corrections.get(verb, verb)

def create_graph(text):
    doc = nlp(text)
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
        notebook=False
    )
    
    added_nodes = set()
    added_edges = set()
    entity_map = {ent.text: ent.label_ for ent in doc.ents}

    def clean_text(text):
        return text.strip().strip(",.!?;:'\"()[]{}").replace('\n', ' ')

    def get_main_entity(text):
        for entity in entity_map:
            if text in entity:
                return entity
        return clean_text(text)

    def get_subj_obj_pairs(sent):
        pairs = []
        verbs = [token for token in sent if token.pos_ == "VERB"]
        
        for verb in verbs:
            # Get subjects (left side of verb)
            subjects = []
            for token in verb.lefts:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    subject = ' '.join([t.text for t in token.subtree])
                    subjects.append(subject)
            
            # Handle passive voice (agent)
            if verb.dep_ == "auxpass":
                for token in verb.head.children:
                    if token.dep_ == "agent":
                        agent = ' '.join([t.text for t in token.subtree])
                        subjects.append(agent)
            
            # Get objects (right side of verb)
            objects = []
            for token in verb.rights:
                if token.dep_ in ("dobj", "attr", "obj", "oprd"):
                    obj = ' '.join([t.text for t in token.subtree])
                    objects.append(obj)
                elif token.dep_ == "prep":
                    for child in token.children:
                        if child.dep_ == "pobj":
                            pobj = ' '.join([t.text for t in child.subtree])
                            objects.append(pobj)
            
            # Create subject-verb-object pairs
            for subj in subjects:
                for obj in objects:
                    pairs.append((subj, verb.lemma_, obj))
        
        return pairs

    # Process each sentence
    for sent in doc.sents:
        pairs = get_subj_obj_pairs(sent)
        
        for subj, verb, obj in pairs:
            # Clean and normalize the nodes
            subj = get_main_entity(subj)
            obj = get_main_entity(obj)
            
            if not subj or not obj:
                continue
                
            # Add nodes if not already present
            if subj not in added_nodes:
                net.add_node(
                    subj,
                    label=subj,
                    color=get_color_for_entity(subj, entity_map),
                    size=20
                )
                added_nodes.add(subj)
                
            if obj not in added_nodes:
                net.add_node(
                    obj,
                    label=obj,
                    color=get_color_for_entity(obj, entity_map),
                    size=20
                )
                added_nodes.add(obj)
            
            # Add edge if not already present
            edge = (subj, obj, verb)
            if edge not in added_edges:
                net.add_edge(
                    subj,
                    obj,
                    label=fix_verb_label(verb),
                    width=2
                )
                added_edges.add(edge)

    # Save the graph
    graph_filename = "knowledge_graph.html"
    graph_path = os.path.join(app.config['STATIC_FOLDER'], graph_filename)
    
    # Use write_html instead of show
    net.write_html(graph_path)
    
    return graph_filename  # Return just the filename for template rendering

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

@app.route('/process', methods=['POST'])
def process():
    input_type = request.form['input_type']
    text = ""

    if input_type == 'text':
        text = request.form['text_input']
    elif input_type == 'url':
        url = request.form['url_input']
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
    elif input_type == 'pdf':
        pdf_file = request.files['pdf_file']
        if pdf_file:
            pdf_path = os.path.join("uploads", pdf_file.filename)
            pdf_file.save(pdf_path)
            text = extract_text_from_pdf(pdf_path)

    # Extract entities using spaCy
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Generate graph and save it to static/graph.html
    # graph_path = create_graph(entities)
    graph_path = create_graph(text)

    # Render the index.html with the entities and the graph
    return render_template('index.html', entities=entities, graph_path=graph_path)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_type = request.form.get('input_type')
        text = ""
        error = None
        
        try:
            if input_type == 'text':
                text = request.form.get('text_input', '')
            elif input_type == 'url':
                url = request.form.get('url_input', '')
                if url:
                    article = Article(url)
                    article.download()
                    article.parse()
                    text = article.text
                else:
                    error = "URL cannot be empty"
            elif input_type == 'pdf':
                pdf_file = request.files.get('pdf_file')
                if pdf_file and pdf_file.filename:
                    filename = secure_filename(pdf_file.filename)
                    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    pdf_file.save(pdf_path)
                    text = extract_text_from_pdf(pdf_path)
                else:
                    error = "No PDF file selected"
            
            if text and not error:
                graph_filename = create_graph(text)
                doc = nlp(text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                return render_template(
                    'index.html',
                    entities=entities,
                    graph_filename=graph_filename,
                    success=True
                )
            else:
                return render_template('index.html', error=error or "No text input provided")
        
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
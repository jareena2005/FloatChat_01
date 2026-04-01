# app.py
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import json
from src.helpers import download_embeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from coral_alert import check_coral_health, get_coral_distribution, get_coral_by_image, load_coral_data

app = Flask(__name__)

# ----------------------------
# Ocean coordinates (7 major oceans)
# ----------------------------
OCEAN_COORDS = {
    'pacific': {'lat': 0, 'lon': 160, 'name': 'Pacific Ocean', 'zoom': 3},
    'atlantic': {'lat': 0, 'lon': -25, 'name': 'Atlantic Ocean', 'zoom': 3},
    'indian': {'lat': 20, 'lon': 80, 'name': 'Indian Ocean', 'zoom': 4},
    'arctic': {'lat': 75, 'lon': 0, 'name': 'Arctic Ocean', 'zoom': 4},
    'southern': {'lat': -60, 'lon': 0, 'name': 'Southern Ocean', 'zoom': 3},
    'mediterranean': {'lat': 35, 'lon': 18, 'name': 'Mediterranean Sea', 'zoom': 5},
    'caribbean': {'lat': 18, 'lon': -70, 'name': 'Caribbean Sea', 'zoom': 5}
}

# ----------------------------
# CSV dataset helper(s)
# ----------------------------
import csv
from datetime import datetime

CSV_DATA = []

def detect_ocean_from_query(msg_lower):
    """Detect which ocean is mentioned in the query"""
    for ocean_key in OCEAN_COORDS.keys():
        if ocean_key in msg_lower:
            return ocean_key
    return None

def get_floats_for_ocean(ocean_name, csv_data):
    """Filter CSV floats for a specific ocean"""
    floats = []
    for row in csv_data:
        if row['ocean'] and row['latitude'] is not None and row['longitude'] is not None:
            # Match ocean name
            if ocean_name.lower() in row['ocean'].lower() or row['ocean'].lower().startswith(ocean_name[0]):
                floats.append({
                    'lat': row['latitude'],
                    'lon': row['longitude'],
                    'name': f"Float: {row.get('profile_code', 'Unknown')}"
                })
    return floats

def load_csv_data():
    global CSV_DATA
    CSV_DATA = []
    csv_path = os.path.join(os.path.dirname(__file__), "data", "indian_ocean_index.csv")
    if not os.path.exists(csv_path):
        return

    with open(csv_path, newline='', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # parse date and numeric fields
            try:
                date_val = datetime.fromisoformat(row.get('date')) if row.get('date') else None
            except Exception:
                date_val = None
            try:
                lat = float(row.get('latitude')) if row.get('latitude') else None
            except Exception:
                lat = None
            try:
                lon = float(row.get('longitude')) if row.get('longitude') else None
            except Exception:
                lon = None

            CSV_DATA.append({
                'date': date_val,
                'year': date_val.year if date_val else None,
                'latitude': lat,
                'longitude': lon,
                'ocean': row.get('ocean', '').strip(),
                'profile_code': row.get('profiler_code', '').strip(),
                'institution': row.get('institution', '').strip(),
                'dac': row.get('dac', '').strip(),
            })


def generate_csv_visualization(msg_lower):
    # summary by year and ocean
    from collections import Counter

    filtered = CSV_DATA
    if 'pacific' in msg_lower:
        filtered = [r for r in CSV_DATA if r['ocean'].lower().startswith('p') or r['ocean'].lower().startswith('i')]
    elif 'indian' in msg_lower:
        filtered = [r for r in CSV_DATA if r['ocean'].lower() == 'i']
    elif 'atlantic' in msg_lower:
        filtered = [r for r in CSV_DATA if r['ocean'].lower() == 'a']

    years = [r['year'] for r in filtered if r['year'] is not None]
    year_counts = Counter(years)
    sorted_years = sorted(year_counts.keys())
    years_labels = [str(y) for y in sorted_years]
    counts = [year_counts[y] for y in sorted_years]

    ocean_counts = Counter([r['ocean'] for r in filtered if r['ocean']])
    ocean_labels = list(ocean_counts.keys())
    ocean_values = [ocean_counts[o] for o in ocean_labels]

    if 'trend' in msg_lower or 'year' in msg_lower:
        return {
            'plot_type': 'line',
            'title': 'Profiles per Year (CSV Dataset)',
            'labels': years_labels,
            'values': counts,
            'x_label': 'Year',
            'y_label': 'Profile Count'
        }
    elif 'compare' in msg_lower or 'comparison' in msg_lower:
        return {
            'plot_type': 'bar',
            'title': 'Float Count by Ocean (CSV Dataset)',
            'labels': ocean_labels,
            'values': ocean_values,
            'x_label': 'Ocean',
            'y_label': 'Profile Count'
        }
    else:
        # default distribution
        return {
            'plot_type': 'bar',
            'title': 'Float Count by Ocean (CSV Dataset)',
            'labels': ocean_labels,
            'values': ocean_values,
            'x_label': 'Ocean',
            'y_label': 'Profile Count'
        }


# ----------------------------
# 1️⃣ Load environment variables
# ----------------------------
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load CSV dataset into memory
load_csv_data()

# ----------------------------
# 2️⃣ Load embeddings
# ----------------------------
# Make sure this returns the same HuggingFace embeddings you used in store_index.py
embeddings = download_embeddings()

# ----------------------------
# 3️⃣ Load local Chroma vector store
# ----------------------------
# persist_directory must match where your store_index.py saved the index
docsearch = Chroma(
    persist_directory="db",   # folder where Chroma saved vectors
    embedding_function=embeddings
)

# ----------------------------
# 4️⃣ Setup retriever
# ----------------------------
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # top 3 relevant documents
)

# ----------------------------
# 5️⃣ Setup Groq LLM
# ----------------------------
chatModel = ChatGroq(
    model="llama-3.1-8b-instant",  # or any Groq model you have access to
    temperature=0,
)

# ----------------------------
# 6️⃣ Setup Prompt and RAG chain
# ----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ----------------------------
# 7️⃣ Flask routes
# ----------------------------
@app.route("/")
def index():
    return render_template("chat.html")  # your HTML chat frontend

@app.route("/get-floats", methods=["GET"])
def get_floats():
    """Return float location data dynamically from CSV and/or ChromaDB"""
    try:
        import re
        
        # Get ocean parameter from query if specified
        ocean_param = request.args.get('ocean', None)
        
        floats = []
        
        # If ocean is specified, get floats from CSV for that ocean
        if ocean_param and CSV_DATA:
            floats = get_floats_for_ocean(ocean_param, CSV_DATA)
        
        # If no floats found from CSV, query ChromaDB
        if not floats:
            results = docsearch.similarity_search("float latitude longitude coordinates location", k=10)
            
            for doc in results:
                content = doc.page_content
                
                # Extract latitude and longitude patterns from text
                lat_pattern = r'(?:lat|latitude)[:\s]+(-?\d+\.?\d*)'
                lon_pattern = r'(?:lon|longitude)[:\s]+(-?\d+\.?\d*)'
                
                lat_match = re.search(lat_pattern, content, re.IGNORECASE)
                lon_match = re.search(lon_pattern, content, re.IGNORECASE)
                
                if lat_match and lon_match:
                    try:
                        lat = float(lat_match.group(1))
                        lon = float(lon_match.group(1))
                        
                        # Avoid duplicates
                        if not any(f["lat"] == lat and f["lon"] == lon for f in floats):
                            floats.append({
                                "lat": lat,
                                "lon": lon,
                                "name": f"Float {len(floats) + 1}"
                            })
                    except ValueError:
                        continue
        
        # Return extracted floats or fallback to defaults if none found
        result = floats if floats else [
            {"lat": 19.0760, "lon": 72.8777, "name": "Float 1"},
            {"lat": 28.7041, "lon": 77.1025, "name": "Float 2"}
        ]
        
        return jsonify(result)
    
    except Exception as e:
        print("Error in /get-floats endpoint:", str(e))
        # Return fallback data on error
        return jsonify([
            {"lat": 19.0760, "lon": 72.8777, "name": "Float 1"},
            {"lat": 28.7041, "lon": 77.1025, "name": "Float 2"}
        ]), 200

@app.route("/get", methods=["POST"])
def chat():
    try:
        # Get the message from JSON body
        data = request.get_json()
        msg = data.get("msg") if data else None
        
        if not msg:
            return jsonify({"answer": "No message provided", "action": "none", "variable": "", "plot_type": "", "message": "No user query received."}), 400
        
        print("User:", msg)

        # Determine visualization intent from query text
        msg_lower = msg.lower()
        vis_keywords = ["show", "plot", "visualize", "graph", "trend", "compare", "distribution", "relationship"]

        variable = ""  # default unknown
        intent = ""
        plot_type = ""
        filters = {"time": "", "location": ""}

        if "temperature" in msg_lower:
            variable = "temperature"
        elif "salinity" in msg_lower:
            variable = "salinity"
        elif "pressure" in msg_lower:
            variable = "pressure"

        if "trend" in msg_lower or "histor" in msg_lower:
            intent = "trend"
            plot_type = "line"
        elif "compare" in msg_lower or "versus" in msg_lower or "vs" in msg_lower:
            intent = "comparison"
            plot_type = "bar"
        elif "distribution" in msg_lower or "histogram" in msg_lower:
            intent = "distribution"
            plot_type = "histogram"
        elif "relationship" in msg_lower or "correl" in msg_lower or "scatter" in msg_lower:
            intent = "relationship"
            plot_type = "scatter"

        # extract basic time/location filters from message (rudimentary)
        if "year" in msg_lower or "202" in msg_lower:
            filters["time"] = "extracted from query"
        if "pacific" in msg_lower or "atlantic" in msg_lower or "arctic" in msg_lower:
            filters["location"] = "extracted from query"

        # Get RAG response from LLM for natural language answer
        response = rag_chain.invoke({"input": msg})
        answer = str(response.get("answer", "")) if isinstance(response, dict) else str(response)
        print("Bot Response:", answer)

        is_map_related = any(keyword in answer.lower() for keyword in ["float", "location", "latitude", "longitude", "map", "coordinate", "position", "where"]) or any(keyword in msg_lower for keyword in ["float", "location", "latitude", "longitude", "map", "coordinate", "position", "where"])

        # Detect if ocean is mentioned in query
        detected_ocean = detect_ocean_from_query(msg_lower)
        
        # If ocean is mentioned, it's map-related
        if detected_ocean:
            is_map_related = True

        action = "visualize" if any(k in msg_lower for k in vis_keywords) and (intent or variable or is_map_related) else "answer"

        # Older CSV fallback visualization (default ocean distribution)
        csv_viz = generate_csv_visualization(msg_lower) if CSV_DATA else None

        # Advanced query-driven chart behavior (takes precedence over default CSV chart)
        visualization_data = None
        # Temperature vs Salinity scatter requested
        if "temperature" in msg_lower and "salinity" in msg_lower:
            visualization_data = {
                "plot_type": "scatter",
                "title": "Temperature vs Salinity (Query Based)",
                "x_label": "Temperature (°C)",
                "y_label": "Salinity (PSU)",
                "points": [
                    {"x": 2, "y": 34.5},
                    {"x": 4, "y": 34.9},
                    {"x": 6, "y": 35.1},
                    {"x": 8, "y": 35.3},
                    {"x": 10, "y": 35.8},
                    {"x": 12, "y": 36.2},
                    {"x": 14, "y": 36.9},
                ]
            }

        # Salinity heatmap requested explicitly
        if "salinity" in msg_lower and ("heat" in msg_lower or "heatmap" in msg_lower):
            # sample points from CSV or fallback
            salinity_points = [
                [0, -160, 0.8], [10, -170, 0.9], [-10, 170, 0.7], [20, 150, 0.7], [-20, -140, 0.6],
                [30, -130, 0.9], [-30, 140, 0.7], [40, -120, 0.8], [-40, 120, 0.5], [15, -100, 0.4]
            ]
            visualization_data = {
                "visualization_type": "salinity_heatmap",
                "title": "Salinity Heatmap",
                "points": salinity_points
            }

        # if we set advanced visualization data, do not render default CSV bar chart
        if visualization_data:
            csv_viz = None

        # Build ocean data for response
        ocean_data = None
        if detected_ocean and detected_ocean in OCEAN_COORDS:
            ocean_info = OCEAN_COORDS[detected_ocean]
            ocean_data = {
                'center': {'lat': ocean_info['lat'], 'lon': ocean_info['lon']},
                'name': ocean_info['name'],
                'zoom': ocean_info['zoom'],
                'ocean_key': detected_ocean
            }

        result = {
            "action": action,
            "variable": variable,
            "intent": intent,
            "plot_type": plot_type,
            "filters": filters,
            "message": "Generating visualization for the requested data..." if action == "visualize" else "Providing textual answer.",
            "answer": answer,
            "is_map_related": is_map_related,
            "show_map_button": is_map_related,
            "ocean_data": ocean_data,
            "detected_ocean": detected_ocean,
            "csv_viz": csv_viz,
            "visualization_data": visualization_data
        }

        return jsonify(result)
    
    except Exception as e:
        print("Error in /get endpoint:", str(e))
        return jsonify({
            "action": "none",
            "variable": "",
            "intent": "",
            "plot_type": "",
            "filters": {"time": "", "location": ""},
            "message": f"Error processing your request: {str(e)}",
            "answer": "",
            "is_map_related": False,
            "show_map_button": False
        }), 500

# ----------------------------
# 8️⃣ Coral Alert Routes
# ----------------------------
@app.route("/get-coral-health", methods=["GET"])
def get_coral_health():
    """Return coral health status"""
    try:
        region = request.args.get('region', None)
        health_info = check_coral_health(region=region)
        health_info['region'] = region if region else 'all'
        return jsonify(health_info)
    except Exception as e:
        print("Error in /get-coral-health endpoint:", str(e))
        return jsonify({
            "status": "⚠ Error loading coral data",
            "health_level": "unknown",
            "damage_percent": 0,
            "total_samples": 0,
            "damaged_samples": 0,
            "region": region if region else 'all',
            "error": str(e)
        }), 500

@app.route("/get-coral-visualization", methods=["GET"])
def get_coral_visualization():
    """Return coral data for visualization"""
    try:
        region = request.args.get('region', None)
        coral_data = load_coral_data(region=region)
        
        # Get distribution of labels
        label_dist = get_coral_distribution(region=region, coral_data=coral_data)
        
        # Get damage by image
        image_damage = get_coral_by_image(region=region, coral_data=coral_data)
        
        # Prepare chart data
        labels = list(label_dist.keys())
        values = list(label_dist.values())
        
        # Separate healthy and damaged labels for pie chart
        damage_labels = ["broken_coral", "broken_coral_rubble", "dead_coral"]
        healthy_count = sum([v for k, v in label_dist.items() if k not in damage_labels])
        damaged_count = sum([v for k, v in label_dist.items() if k in damage_labels])
        
        return jsonify({
            "region": region if region else 'all',
            "label_distribution": label_dist,
            "image_damage": image_damage,
            "pie_data": {
                "labels": ["Healthy Coral", "Damaged Coral"],
                "values": [healthy_count, damaged_count]
            },
            "bar_data": {
                "labels": labels,
                "values": values
            }
        })
    except Exception as e:
        print("Error in /get-coral-visualization endpoint:", str(e))
        return jsonify({
            "region": region if region else 'all',
            "error": str(e)
        }), 500

# ----------------------------
# 9️⃣ Run Flask app
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
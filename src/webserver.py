import os
import logging
from flask import Flask, render_template, request, jsonify

from search import (
    load_models_and_db,
    enhance_query,
    query_chroma_db,
    generate_content_summary,
    suggest_next_queries
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Global variables to store models and database
vector_store = None
llm = None


@app.route('/')
def index():
    """Render the main chat page."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data.'}), 400

        user_query = data.get('query', '').strip()
        raw_mode = data.get('raw_mode', False)

        if not user_query:
            return jsonify({'error': 'Please enter a valid query.'}), 400

        if not raw_mode:
            enhanced_query = enhance_query(llm, user_query)
        else:
            enhanced_query = user_query

        results = query_chroma_db(vector_store, enhanced_query, num_results=1)

        if not results:
            return jsonify({
                'response': 'No matching results found.',
                'suggestions': []
            })

        result = results[0]

        if raw_mode:
            # Simple response for raw mode
            response = f"""**Raw Results for: '{user_query}'**

**Score:** {result['similarity_score']:.4f}

**Content:** {result['content']}

**Metadata:** {result['metadata']}"""
            suggestions = []
        else:
            content_summary = generate_content_summary(llm, result["content"], user_query)
            content_preview = result["content"]
            if len(content_preview) > 500:
                content_preview = content_preview[:500] + "..."

            response = f"""**Summary:**
{content_summary}

**Relevance Score:** {result['similarity_score']:.4f}

**Content Preview:**
{content_preview}

**Metadata:** {result['metadata']}"""
            # Generate suggestions using your existing function
            suggestions = suggest_next_queries(llm, result["content"], user_query)

        return jsonify({
            'response': response,
            'suggestions': suggestions
        })

    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request.'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found.'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error.'}), 500


def initialize_app():
    global vector_store, llm

    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(os.path.dirname(current_dir), "database", "chroma_db")

    try:
        vector_store, llm = load_models_and_db(db_path)
        logging.info("initialized successfully")
    except Exception as e:
        logging.error(f"Failed : {str(e)}")
        raise


if __name__ == '__main__':
    initialize_app()
    app.run(debug=True, host='0.0.0.0', port=1337)
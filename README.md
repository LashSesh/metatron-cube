# Metatron Cube – Post-Symbolic Cognition Engine

**Metatron Cube** is a post-symbolic, cybernetic cognition core that operationalizes the logical and mathematical structure of Metatron’s Cube for the first time as a fully functional prototype.  
This system provides the foundation for a new generation of AI and cognition engines based on emergent, multidimensional logic and field-based operatorics – beyond symbolic/statistical AI.

**Author:** Sebastian Klemm

---

## Features

- Fully documented prototype architecture (Python, Dockerfile)
- API-first design (FastAPI), modular and extensible
- Complete mathematical & algorithmic blueprint for further development (PDF)

---

## Included files

- **/src/** – Python code of the Cognition Core
- **/Dockerfile** – For containerization and easy deployment
- **/docs/METATRON_Blueprint.pdf** – Complete mathematical & algorithmic blueprint
- **/docs/ToE_bySebastianKlemm_v1.0.pdf** – Complete mathematical & algorithmic blueprint
- **requirements.txt** – Python dependencies

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/LashSesh/metatron-cube.git
cd metatron-cube

# Build the Docker image
docker build -t metatron-cube .

# Start the container
docker run -p 8000:8000 metatron-cube

# The API is now accessible at http://localhost:8000

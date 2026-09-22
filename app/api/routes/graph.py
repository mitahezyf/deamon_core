from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from neo4j.exceptions import Neo4jError

from app.api.schemas import SessionMemoryRequest, GraphDataResponse, GraphNode, GraphEdge
from app.core.logger import get_logger

log = get_logger("api.graph")
router = APIRouter()

@router.post("/memory/session", status_code=201)
async def add_session_memory(req: SessionMemoryRequest, request: Request):
    db = request.app.state.db
    if not db or not await db.verify_connectivity():
        raise HTTPException(status_code=503, detail="Memgraph is not available")

    # Upewniamy się, że nazwa relacji nie zawiera niebezpiecznych znaków (tylko alfanumeryczne)
    relation_name = "".join(c for c in req.relation if c.isalnum() or c == "_").upper()
    if not relation_name:
        raise HTTPException(status_code=400, detail="Invalid relation name")

    query = f"""
    MERGE (e:Entity {{name: $entity, session_id: $session_id}})
    MERGE (t:Entity {{name: $target, session_id: $session_id}})
    MERGE (e)-[r:{relation_name}]->(t)
    RETURN e, r, t
    """
    try:
        async with db.driver.session() as session:
            await session.run(
                query, 
                entity=req.entity, 
                target=req.target, 
                session_id=req.session_id
            )
        return {"status": "ok", "message": "Memory saved."}
    except Neo4jError as e:
        log.error(f"Memgraph Error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@router.get("/graph/data", response_model=GraphDataResponse)
async def get_graph_data(request: Request):
    db = request.app.state.db
    if not db or not await db.verify_connectivity():
        raise HTTPException(status_code=503, detail="Memgraph is not available")

    query = "MATCH (n)-[r]->(m) RETURN id(n) AS n_id, n.name AS n_name, id(m) AS m_id, m.name AS m_name, id(r) AS r_id, type(r) AS r_type LIMIT 100"
    
    nodes = {}
    edges = []
    
    try:
        async with db.driver.session() as session:
            result = await session.run(query)
            async for record in result:
                n_id = record["n_id"]
                n_name = record["n_name"]
                m_id = record["m_id"]
                m_name = record["m_name"]
                r_type = record["r_type"]
                
                if n_id not in nodes:
                    nodes[n_id] = GraphNode(id=n_id, label=n_name or "Unknown")
                if m_id not in nodes:
                    nodes[m_id] = GraphNode(id=m_id, label=m_name or "Unknown")
                    
                edges.append(GraphEdge(from_id=n_id, to_id=m_id, label=r_type))
                
        return GraphDataResponse(nodes=list(nodes.values()), edges=edges)
    except Neo4jError as e:
        log.error(f"Memgraph Error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@router.get("/graph/ui", response_class=HTMLResponse)
async def graph_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta charset="UTF-8">
        <title>DAEMON GraphRAG Live</title>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style type="text/css">
            body { font-family: sans-serif; background-color: #1e1e1e; color: #fff; margin: 0; padding: 0;}
            #mynetwork {
                width: 100vw;
                height: 100vh;
                border: 1px solid lightgray;
                box-sizing: border-box;
            }
        </style>
    </head>
    <body>
        <div id="mynetwork"></div>
        <script type="text/javascript">
            var container = document.getElementById('mynetwork');
            var nodes = new vis.DataSet([]);
            var edges = new vis.DataSet([]);
            var data = { nodes: nodes, edges: edges };
            var options = {
                nodes: {
                    shape: 'dot',
                    size: 20,
                    font: { size: 16, color: '#ffffff' },
                    borderWidth: 2,
                    color: { background: '#97C2FC', border: '#2B7CE9' }
                },
                edges: {
                    width: 2,
                    font: { size: 12, color: '#ffffff', align: 'top', strokeWidth: 0 },
                    color: { color: '#848484' },
                    arrows: { to: { enabled: true, scaleFactor: 1 } }
                },
                physics: { stabilization: false }
            };
            var network = new vis.Network(container, data, options);

            async function fetchData() {
                try {
                    const response = await fetch('/graph/data');
                    const json = await response.json();
                    
                    // Przepakowanie krawędzi (zmiana nazwy from_id -> from, to_id -> to)
                    const visEdges = json.edges.map(e => ({
                        from: e.from_id,
                        to: e.to_id,
                        label: e.label
                    }));
                    
                    nodes.update(json.nodes);
                    edges.update(visEdges);
                } catch (error) {
                    console.error("Błąd pobierania danych grafu:", error);
                }
            }

            // Inicjalne pobranie
            fetchData();
            // Polling co 5 sekund
            setInterval(fetchData, 5000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

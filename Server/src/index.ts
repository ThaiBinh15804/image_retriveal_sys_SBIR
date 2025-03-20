import express from 'express'
import axios from 'axios'
import cors from 'cors'

const app = express()
const PORT = 2020
const GRAPHDB_ENDPOINT = 'http://localhost:7200/repositories/new'

app.use(express.json())
app.use(cors())
app.use('/images', express.static('D:/VS_CODE/Python/Flickr/flickr30k_images/flickr30k_images/')) 
app.post('/query', async (req, res) => {
  const body = req.body.body
  try {
    const sparqlResponse = await axios.post("http://localhost:8000/generate_sparql", { text: body })
        
    if (!sparqlResponse.data.sparql_query) {
        throw new Error("Invalid SPARQL response from API.");
    }

    let sparql = sparqlResponse.data.sparql_query;

    sparql = sparql.replace(/\n/g, " ").replace(/\s+/g, " ").trim();
    console.log(sparql)

    const response = await axios.post(GRAPHDB_ENDPOINT, `query=${encodeURIComponent(sparql)}`, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        Accept: 'application/sparql-results+json'
      }
    })

    res.json({
      bindings: response.data.results.bindings,
      sparql_query: sparqlResponse.data.sparql_query
    })
  } catch (error: any) {
    console.error('lỗi truy vấn graphDB:', error.response ? error.response.data : error.message)
    res.status(500).json({ error: 'Lỗi truy vấn graphDB' })
  }
})

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`)
})

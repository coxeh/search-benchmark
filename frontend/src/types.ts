export interface Dataset {
  id: number
  name: string
  filename: string
  created_at: string | null
  row_count: number
}

export interface SearchResult {
  id: number
  score: number
  attributes: Record<string, string | number>
  dataset_name?: string
}

export interface SearchResponse {
  results: SearchResult[]
  latency_ms: number
  method: string
}

export interface CompareResponse {
  vector: { results: SearchResult[]; latency_ms: number; error?: string }
  faiss: { results: SearchResult[]; latency_ms: number; error?: string }
  hybrid: { results: SearchResult[]; latency_ms: number; error?: string }
  trigram: { results: SearchResult[]; latency_ms: number; error?: string }
  ilike: { results: SearchResult[]; latency_ms: number; error?: string }
  fulltext: { results: SearchResult[]; latency_ms: number; error?: string }
}

export interface Stats {
  entity_count: number
  dataset_count: number
  entities_with_embedding: number
  index_sizes: { name: string; size: string }[]
}

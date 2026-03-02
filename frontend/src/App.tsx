import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { Dataset, SearchResponse, CompareResponse, Stats } from './types'

const SEARCH_METHODS = [
  { value: 'vector', label: 'Vector (semantic)' },
  { value: 'faiss', label: 'FAISS' },
  { value: 'hybrid', label: 'Hybrid (vector + full-text)' },
  { value: 'trigram', label: 'Trigram (fuzzy)' },
  { value: 'ilike', label: 'ILIKE (pattern)' },
  { value: 'fulltext', label: 'Full-text' },
  { value: 'compare', label: 'Compare All' },
] as const

type SearchMethod = (typeof SEARCH_METHODS)[number]['value']

async function fetchDatasets(): Promise<Dataset[]> {
  const r = await fetch('/api/datasets')
  if (!r.ok) throw new Error('Failed to fetch datasets')
  return r.json()
}

async function fetchSearch(
  q: string,
  method: string,
  datasetId: number | null,
  limit: number
): Promise<SearchResponse> {
  const params = new URLSearchParams({ q, method, limit: String(limit) })
  if (datasetId) params.set('dataset_id', String(datasetId))
  const r = await fetch(`/api/search?${params}`)
  if (!r.ok) throw new Error('Search failed')
  return r.json()
}

async function fetchCompare(
  q: string,
  datasetId: number | null,
  limit: number
): Promise<CompareResponse> {
  const params = new URLSearchParams({ q, limit: String(limit) })
  if (datasetId) params.set('dataset_id', String(datasetId))
  const r = await fetch(`/api/search/compare?${params}`)
  if (!r.ok) throw new Error('Compare failed')
  return r.json()
}

async function fetchStats(): Promise<Stats> {
  const r = await fetch('/api/stats')
  if (!r.ok) throw new Error('Failed to fetch stats')
  return r.json()
}

function formatLabel(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Split query into terms for matching (min 2 chars, escape regex). */
function getQueryTerms(query: string): string[] {
  return query
    .trim()
    .split(/\s+/)
    .filter((t) => t.length >= 2)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
}

/** Check if value contains any query term (case-insensitive). */
function valueMatches(value: string, terms: string[]): boolean {
  if (!terms.length) return false
  const lower = value.toLowerCase()
  return terms.some((t) => lower.includes(t.toLowerCase()))
}

/** Render text with matching substrings wrapped in <mark>. */
function HighlightedValue({ text, query }: { text: string; query: string }) {
  const terms = getQueryTerms(query)
  if (!terms.length) return <span className="truncate text-slate-200">{text}</span>

  const regex = new RegExp(`(${terms.join('|')})`, 'gi')
  const parts = text.split(regex)
  const isMatch = (part: string) => terms.some((t) => new RegExp(`^${t}$`, 'i').test(part))

  return (
    <span className="truncate text-slate-200">
      {parts.map((part, i) =>
        isMatch(part) ? (
          <mark key={i} className="rounded bg-amber-500/40 px-0.5 text-amber-100">
            {part}
          </mark>
        ) : (
          part
        )
      )}
    </span>
  )
}

function ResultCard({
  result,
  query = '',
  showScore = true,
  compact = false,
}: {
  result: { id: number; score: number; attributes: Record<string, unknown>; dataset_name?: string }
  query?: string
  showScore?: boolean
  compact?: boolean
}) {
  const [embeddingText, setEmbeddingText] = useState<string | null>(null)
  const [loadingEmbedding, setLoadingEmbedding] = useState(false)
  const [showEmbedding, setShowEmbedding] = useState(false)

  const scorePct = Math.round(result.score * 100)
  const terms = getQueryTerms(query)

  const fetchEmbeddingText = async () => {
    if (embeddingText !== null) {
      setShowEmbedding((s) => !s)
      return
    }
    setLoadingEmbedding(true)
    try {
      let r = await fetch(`/api/entities/${result.id}/search-text`)
      if (r.status === 404) {
        r = await fetch('/api/search-text/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ attributes: result.attributes }),
        })
      }
      if (!r.ok) throw new Error('Failed to fetch')
      const data = await r.json()
      setEmbeddingText(data.search_text ?? '')
      setShowEmbedding(true)
    } catch {
      setEmbeddingText('(Could not load embedding text)')
      setShowEmbedding(true)
    } finally {
      setLoadingEmbedding(false)
    }
  }

  // Sort: matched columns first
  const entries = Object.entries(result.attributes)
  const sortedEntries =
    terms.length > 0
      ? [...entries].sort((a, b) => {
          const aMatch = valueMatches(String(a[1]), terms)
          const bMatch = valueMatches(String(b[1]), terms)
          if (aMatch && !bMatch) return -1
          if (!aMatch && bMatch) return 1
          return 0
        })
      : entries

  return (
    <div
      className={`rounded-lg border border-slate-700 bg-slate-800/50 p-4 ${
        compact ? 'text-sm' : ''
      }`}
    >
      <div className="mb-2 flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2 flex-wrap">
          {showScore && (
            <div
              className="flex items-center gap-2"
              title="Relevance score: higher % = better match to your query"
            >
              <div className="h-2 w-20 overflow-hidden rounded-full bg-slate-700">
                <div
                  className="h-full bg-emerald-500"
                  style={{ width: `${scorePct}%` }}
                />
              </div>
              <span className="text-emerald-400 tabular-nums">{scorePct}%</span>
            </div>
          )}
          <button
            type="button"
            onClick={fetchEmbeddingText}
            disabled={loadingEmbedding}
            className="rounded bg-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-600 disabled:opacity-50"
          >
            {loadingEmbedding
              ? 'Loading...'
              : showEmbedding && embeddingText !== null
                ? 'Hide embedding text'
                : 'View embedding text'}
          </button>
        </div>
        {result.dataset_name && (
          <span className="rounded bg-slate-700 px-2 py-0.5 text-xs text-slate-400">
            {result.dataset_name}
          </span>
        )}
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-1">
        {sortedEntries.map(([key, value]) => {
          const strValue = String(value)
          const isMatched = valueMatches(strValue, terms)
          return (
            <div
              key={key}
              className={`flex gap-2 ${isMatched ? 'rounded bg-amber-500/10 px-2 py-0.5 -mx-2 -my-0.5' : ''}`}
            >
              <span
                className={`shrink-0 ${isMatched ? 'font-medium text-amber-400' : 'text-slate-500'}`}
              >
                {formatLabel(key)}:
              </span>
              <HighlightedValue text={strValue} query={query} />
            </div>
          )
        })}
      </div>
      {showEmbedding && embeddingText !== null && (
        <div className="mt-3 rounded bg-slate-700/50 px-3 py-2 text-xs text-slate-400">
          <p className="mb-1 font-medium text-slate-300">Text used for embedding:</p>
          <p className="whitespace-pre-wrap break-words">{embeddingText || '(empty)'}</p>
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [submittedQuery, setSubmittedQuery] = useState('')
  const [method, setMethod] = useState<SearchMethod>('vector')
  const [datasetId, setDatasetId] = useState<number | null>(null)
  const [limit] = useState(20)

  const { data: datasets = [], isLoading: datasetsLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: fetchDatasets,
  })

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
    refetchInterval: 10_000,
  })

  const { data: searchData, isLoading: searchLoading, isFetching: searchFetching } = useQuery({
    queryKey: ['search', submittedQuery, method, datasetId, limit],
    queryFn: () =>
      method === 'compare'
        ? fetchCompare(submittedQuery, datasetId, 10)
        : fetchSearch(submittedQuery, method, datasetId, limit),
    enabled: submittedQuery.length > 0,
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) setSubmittedQuery(query.trim())
  }

  const isCompare = method === 'compare'
  const compareData = isCompare ? (searchData as CompareResponse | undefined) : null
  const singleData = !isCompare ? (searchData as SearchResponse | undefined) : null

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <div className="mx-auto max-w-6xl px-4 py-8">
        <header className="mb-8">
          <h1 className="text-2xl font-bold text-white">Vector Search</h1>
          <p className="mt-1 text-slate-400">
            Semantic and text search over supplier data (PostgreSQL + pgvector + BGE-M3)
          </p>
        </header>

        {stats && (
          <div className="mb-6 flex flex-wrap gap-4 rounded-lg bg-slate-800/50 px-4 py-3 text-sm">
            <span>
              <strong>{stats.entity_count}</strong> entities
            </span>
            <span>
              <strong>{stats.dataset_count}</strong> datasets
            </span>
            <span>
              <strong>{stats.entities_with_embedding}</strong> with embeddings
            </span>
          </div>
        )}

        <div className="mb-4 rounded-lg border border-slate-700 bg-slate-800/30 px-4 py-3 text-sm text-slate-400">
          <p className="font-medium text-slate-300">Search tips</p>
          <ul className="mt-1 list-inside list-disc space-y-0.5">
            <li>Use 2–5 key terms or a short phrase (e.g. <em>ISO certified metal supplier</em>)</li>
            <li>Vector and hybrid work best with descriptive queries; full-text and trigram match exact words</li>
            <li>Terms under 2 characters are ignored for result highlighting</li>
            <li><strong>Score bar:</strong> relevance to your query — higher % means a better match</li>
          </ul>
        </div>

        <form onSubmit={handleSubmit} className="mb-8 flex flex-wrap gap-3">
          <select
            value={datasetId ?? ''}
            onChange={(e) => setDatasetId(e.target.value ? Number(e.target.value) : null)}
            className="rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-slate-200"
            disabled={datasetsLoading}
          >
            <option value="">All datasets</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.row_count})
              </option>
            ))}
          </select>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as SearchMethod)}
            className="rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-slate-200"
          >
            {SEARCH_METHODS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search suppliers..."
            className="min-w-[200px] flex-1 rounded-lg border border-slate-600 bg-slate-800 px-4 py-2 text-slate-200 placeholder-slate-500 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
          />
          <button
            type="submit"
            disabled={searchFetching}
            className="rounded-lg bg-emerald-600 px-6 py-2 font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
          >
            {searchFetching ? 'Searching...' : 'Search'}
          </button>
        </form>

        {searchLoading && submittedQuery && (
          <div className="text-slate-400">Loading...</div>
        )}

        {!searchLoading && singleData && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <span>{singleData.results.length} results</span>
              <span className="rounded bg-slate-700 px-2 py-0.5">
                {singleData.latency_ms.toFixed(1)} ms
              </span>
            </div>
            <div className="flex flex-col gap-4">
              {singleData.results.map((r) => (
                <ResultCard key={r.id} result={r} query={submittedQuery} showScore />
              ))}
            </div>
          </div>
        )}

        {!searchLoading && compareData && (
          <div className="space-y-6">
            <h2 className="text-lg font-semibold">Compare methods</h2>
            <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-5">
              {(['vector', 'faiss', 'hybrid', 'trigram', 'ilike', 'fulltext'] as const).map((m) => {
                const block = compareData[m]
                const err = 'error' in block ? block.error : null
                const results = 'results' in block ? block.results : []
                const latency = 'latency_ms' in block ? block.latency_ms : 0
                return (
                  <div key={m} className="rounded-lg border border-slate-700 bg-slate-800/30 p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <h3 className="font-medium capitalize">{m}</h3>
                      <span className="text-xs text-slate-500">{latency.toFixed(0)} ms</span>
                    </div>
                    {err && <p className="text-sm text-red-400">{err}</p>}
                    <div className="space-y-3">
                      {results.slice(0, 5).map((r) => (
                        <ResultCard
                          key={r.id}
                          result={r}
                          query={submittedQuery}
                          showScore
                          compact
                        />
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {submittedQuery && !searchData && !searchLoading && (
          <p className="text-slate-500">No results or an error occurred.</p>
        )}
      </div>
    </div>
  )
}

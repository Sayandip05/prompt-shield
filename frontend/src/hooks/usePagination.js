import { useState, useCallback } from 'react'

/**
 * Simple pagination hook.
 * @param {number} initialPage
 * @param {number} pageSize
 */
export function usePagination(initialPage = 1, pageSize = 10) {
  const [page, setPage] = useState(initialPage)
  const nextPage = useCallback(() => setPage(p => p + 1), [])
  const prevPage = useCallback(() => setPage(p => Math.max(1, p - 1)), [])
  const resetPage = useCallback(() => setPage(initialPage), [initialPage])
  return { page, pageSize, setPage, nextPage, prevPage, resetPage }
}

export default usePagination

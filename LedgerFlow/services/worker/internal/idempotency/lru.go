// Package idempotency remembers recently processed event IDs.
//
// This is a fast path, not the guarantee: the API deduplicates on event_id
// (processed_events table) and is the source of truth. The cache only saves
// a download and a paid model call when the broker redelivers a message this
// replica has already finished, e.g. after an ack was lost on reconnect.
package idempotency

import (
	"container/list"
	"sync"
)

// LRU is a fixed-capacity set of event IDs with least-recently-used
// eviction. It is safe for concurrent use.
type LRU struct {
	mu       sync.Mutex
	capacity int
	order    *list.List // front = most recently used; values are string IDs
	items    map[string]*list.Element
}

// NewLRU returns a cache holding at most capacity IDs (minimum 1).
func NewLRU(capacity int) *LRU {
	capacity = max(capacity, 1)
	return &LRU{
		capacity: capacity,
		order:    list.New(),
		items:    make(map[string]*list.Element, capacity),
	}
}

// Seen reports whether id was recorded, refreshing its recency if so.
func (c *LRU) Seen(id string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.items[id]
	if ok {
		c.order.MoveToFront(el)
	}
	return ok
}

// Add records id, evicting the least recently used entry when full.
func (c *LRU) Add(id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[id]; ok {
		c.order.MoveToFront(el)
		return
	}
	c.items[id] = c.order.PushFront(id)
	if c.order.Len() > c.capacity {
		oldest := c.order.Back()
		c.order.Remove(oldest)
		delete(c.items, oldest.Value.(string))
	}
}

// Len returns the number of cached IDs.
func (c *LRU) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.order.Len()
}

package idempotency

import (
	"fmt"
	"sync"
	"testing"
)

func TestLRUEvictsLeastRecentlyUsed(t *testing.T) {
	c := NewLRU(2)
	c.Add("a")
	c.Add("b")
	if !c.Seen("a") { // refreshes a, so b is now the eviction candidate
		t.Fatal("a should be cached")
	}
	c.Add("c")

	for id, want := range map[string]bool{"a": true, "b": false, "c": true} {
		if got := c.Seen(id); got != want {
			t.Errorf("Seen(%q) = %v, want %v", id, got, want)
		}
	}
	if c.Len() != 2 {
		t.Errorf("Len = %d, want 2", c.Len())
	}
}

func TestLRUAddIsIdempotent(t *testing.T) {
	c := NewLRU(3)
	c.Add("a")
	c.Add("a")
	if c.Len() != 1 {
		t.Fatalf("Len = %d, want 1", c.Len())
	}
}

func TestLRUConcurrentUse(t *testing.T) {
	c := NewLRU(64)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				id := fmt.Sprintf("%d-%d", g, i%100)
				c.Add(id)
				c.Seen(id)
			}
		}(g)
	}
	wg.Wait()
	if c.Len() > 64 {
		t.Fatalf("Len = %d exceeds capacity", c.Len())
	}
}

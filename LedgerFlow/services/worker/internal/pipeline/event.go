package pipeline

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

const eventTypeDocumentUploaded = "document.uploaded"

// Event is the document.uploaded message from docs/CONTRACTS.md.
type Event struct {
	EventID     string    `json:"event_id"`
	EventType   string    `json:"event_type"`
	OccurredAt  time.Time `json:"occurred_at"`
	TenantID    string    `json:"tenant_id"`
	DocumentID  string    `json:"document_id"`
	StorageKey  string    `json:"storage_key"`
	ContentType string    `json:"content_type"`
	SHA256      string    `json:"sha256"`
	Attempt     int       `json:"attempt"`
	TraceParent string    `json:"trace_parent"`
}

func decodeEvent(body []byte) (Event, error) {
	var ev Event
	if err := json.Unmarshal(body, &ev); err != nil {
		return Event{}, fmt.Errorf("decode event: %w", err)
	}
	if err := ev.validate(); err != nil {
		return Event{}, fmt.Errorf("invalid event %q: %w", ev.EventID, err)
	}
	return ev, nil
}

// validate rejects events that cannot be processed safely. In particular the
// storage key must be the one derived from tenant, document and digest: the
// worker reads with credentials that can see every tenant's objects, so it
// must never follow a key that points elsewhere.
func (ev Event) validate() error {
	var errs []error
	if ev.EventType != eventTypeDocumentUploaded {
		errs = append(errs, fmt.Errorf("unexpected event_type %q", ev.EventType))
	}
	for _, id := range []struct{ name, value string }{
		{"event_id", ev.EventID},
		{"tenant_id", ev.TenantID},
		{"document_id", ev.DocumentID},
	} {
		if !isUUID(id.value) {
			errs = append(errs, fmt.Errorf("%s %q is not a uuid", id.name, id.value))
		}
	}
	if len(ev.SHA256) != 64 || !isLowerHex(ev.SHA256) {
		errs = append(errs, fmt.Errorf("sha256 %q is not a lowercase hex digest", ev.SHA256))
	}
	if want := fmt.Sprintf("tenants/%s/documents/%s/%s", ev.TenantID, ev.DocumentID, ev.SHA256); ev.StorageKey != want {
		errs = append(errs, fmt.Errorf("storage_key %q does not match tenant/document/sha256", ev.StorageKey))
	}
	if ev.ContentType == "" {
		errs = append(errs, errors.New("content_type is empty"))
	}
	return errors.Join(errs...)
}

// isUUID checks the canonical 8-4-4-4-12 textual form in lowercase, which is
// what the API emits.
func isUUID(s string) bool {
	if len(s) != 36 {
		return false
	}
	for i := 0; i < len(s); i++ {
		switch i {
		case 8, 13, 18, 23:
			if s[i] != '-' {
				return false
			}
		default:
			if !isLowerHex(s[i : i+1]) {
				return false
			}
		}
	}
	return true
}

func isLowerHex(s string) bool {
	for i := 0; i < len(s); i++ {
		if c := s[i]; (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

package extract

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"
)

type fakeExtractor struct {
	name  string
	err   error
	calls int
}

func (f *fakeExtractor) Extract(context.Context, Document) (Result, error) {
	f.calls++
	if f.err != nil {
		return Result{}, f.err
	}
	return Result{Extractor: f.name}, nil
}

func TestRouter(t *testing.T) {
	var (
		xmlBody  = []byte("ï»¿  <?xml version=\"1.0\"?><nfeProc/>")
		pdfBody  = []byte("%PDF-1.7\n...")
		errModel = errors.New("model unavailable")
		errRules = errors.New("rules failed")
	)

	tests := []struct {
		name      string
		doc       Document
		rulesErr  error
		modelErr  error
		want      string // extractor that produced the result
		wantErr   error
		wantRules int
		wantModel int
	}{
		{
			name:      "nfe xml uses rules only",
			doc:       Document{ContentType: "application/xml", Content: xmlBody},
			want:      "rules",
			wantRules: 1,
		},
		{
			name:      "content type parameters and case are ignored",
			doc:       Document{ContentType: "Text/XML; charset=UTF-8", Content: xmlBody},
			want:      "rules",
			wantRules: 1,
		},
		{
			name:      "xml that is not nfe goes to model",
			doc:       Document{ContentType: "application/xml", Content: xmlBody},
			rulesErr:  ErrNotApplicable,
			want:      "model",
			wantRules: 1,
			wantModel: 1,
		},
		{
			name:      "broken nfe is not sent to model",
			doc:       Document{ContentType: "application/xml", Content: xmlBody},
			rulesErr:  errRules,
			wantErr:   errRules,
			wantRules: 1,
		},
		{
			name:      "pdf uses model",
			doc:       Document{ContentType: "application/pdf", Content: pdfBody},
			want:      "model",
			wantModel: 1,
		},
		{
			name:      "model failure on pdf is returned for retry",
			doc:       Document{ContentType: "application/pdf", Content: pdfBody},
			modelErr:  errModel,
			wantErr:   errModel,
			wantModel: 1,
		},
		{
			name:      "model failure on mislabelled xml falls back to rules",
			doc:       Document{ContentType: "application/pdf", Content: xmlBody},
			modelErr:  errModel,
			want:      "rules",
			wantModel: 1,
			wantRules: 1,
		},
		{
			name:      "both strategies failing surfaces model error",
			doc:       Document{ContentType: "image/png", Content: xmlBody},
			modelErr:  errModel,
			rulesErr:  ErrNotApplicable,
			wantErr:   errModel,
			wantModel: 1,
			wantRules: 1,
		},
		{
			name:      "octet-stream sniffed as xml uses rules",
			doc:       Document{ContentType: "application/octet-stream", Content: xmlBody},
			want:      "rules",
			wantRules: 1,
		},
		{
			name:    "unsupported type is permanent",
			doc:     Document{ContentType: "application/zip", Content: []byte("PK\x03\x04")},
			wantErr: ErrUnsupportedContentType,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rules := &fakeExtractor{name: "rules", err: tt.rulesErr}
			model := &fakeExtractor{name: "model", err: tt.modelErr}
			r := NewRouter(rules, model, slog.New(slog.NewTextHandler(io.Discard, nil)))

			res, err := r.Extract(context.Background(), tt.doc)

			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("error = %v, want %v", err, tt.wantErr)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if res.Extractor != tt.want {
					t.Fatalf("result from %q, want %q", res.Extractor, tt.want)
				}
			}
			if rules.calls != tt.wantRules || model.calls != tt.wantModel {
				t.Fatalf("calls rules=%d model=%d, want rules=%d model=%d",
					rules.calls, model.calls, tt.wantRules, tt.wantModel)
			}
		})
	}
}

package extract

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
)

// Router picks an extraction strategy per document.
//
// NF-e XML is a signed, schema-defined format: parsing it deterministically
// is exact, free and fast, so rules always go first for XML. The model is
// reserved for formats that have no machine-readable structure (PDF, images)
// and for XML that turns out not to be an NF-e.
type Router struct {
	rules Extractor
	model Extractor
	log   *slog.Logger
}

// NewRouter wires the rule-based and model-based extractors.
func NewRouter(rules, model Extractor, log *slog.Logger) *Router {
	return &Router{rules: rules, model: model, log: log}
}

// Extract implements Extractor.
func (r *Router) Extract(ctx context.Context, doc Document) (Result, error) {
	mt := MediaType(doc.ContentType)
	modelType := isModelMediaType(mt)

	switch {
	case IsXMLMediaType(mt) || (!modelType && LooksLikeXML(doc.Content)):
		res, err := r.rules.Extract(ctx, doc)
		if !errors.Is(err, ErrNotApplicable) {
			return res, err
		}
		r.log.InfoContext(ctx, "xml is not an nf-e; falling back to model", "reason", err)
		return r.model.Extract(ctx, doc)

	case modelType:
		res, err := r.model.Extract(ctx, doc)
		if err == nil || ctx.Err() != nil || !LooksLikeXML(doc.Content) {
			return res, err
		}
		// Uploads are occasionally mislabelled (an NF-e saved as .pdf). The
		// provider rejects or garbles such input, but the rules can read it.
		// If the rules cannot either, surface the model error so the message
		// is retried: that is the failure an operator needs to see.
		ruleRes, ruleErr := r.rules.Extract(ctx, doc)
		if ruleErr != nil {
			return Result{}, fmt.Errorf("model extraction: %w (rule fallback: %w)", err, ruleErr)
		}
		r.log.WarnContext(ctx, "model failed on xml-like content; used rule fallback",
			"content_type", doc.ContentType, "model_error", err)
		return ruleRes, nil

	default:
		return Result{}, fmt.Errorf("%w: %q", ErrUnsupportedContentType, doc.ContentType)
	}
}

// isModelMediaType lists the upload types the contract accepts besides XML.
func isModelMediaType(mt string) bool {
	switch mt {
	case "application/pdf", "image/png", "image/jpeg":
		return true
	default:
		return false
	}
}

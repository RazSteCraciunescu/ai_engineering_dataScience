# ADR 0002: Preserve raw outliers

**Status:** Accepted

Extreme durations remain in raw audit columns and receive an explicit flag. Only the model-specific target is capped. This preserves traceability while limiting the effect of extreme training values.

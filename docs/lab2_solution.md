# Lab 2 – Dynamic Topics + Stored Email Similarity Classification

## Overview

This project extends the existing email topic classification API to support:

- Adding new topics dynamically and persisting them to `data/topic_keywords.json`
- Storing emails with optional ground truth labels in `data/emails.json`
- Dual-mode inference:
  - `mode="topic"`: cosine similarity between email embedding and topic embeddings
  - `mode="email"`: cosine similarity between email embedding and stored email embeddings (nearest neighbor retrieval)

This transforms the system from a static classifier into a configurable and retrieval-enhanced ML system.

---

# Changes Implemented

## 1) Dynamic Topic Creation (`POST /api/v1/topics`)

### What I Changed
- Added a new endpoint to dynamically add or update topics.
- Implemented file persistence using helper functions `read_json()` and `write_json()`.
- Topics are saved in `data/topic_keywords.json`.
- New topics are immediately available for classification without restarting the server.

### Example

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/topics" \
  -H "Content-Type: application/json" \
  -d '{"name":"travel","description":"Flights, hotels, reservations, itinerary changes, trip planning"}'
```

---

## 2) Email Storage with Optional Ground Truth (`POST /api/v1/emails`)

### What I Changed
- Created a new endpoint to store emails.
- Generated embeddings using the existing feature pipeline.
- Stored the following fields in `data/emails.json`:
  - `id` (UUID)
  - `subject`
  - `body`
  - `ground_truth` (optional)
  - `embedding`
  - `created_at` timestamp

### Example

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/emails" \
  -H "Content-Type: application/json" \
  -d '{"subject":"Hotel booking change","body":"Please update my reservation to two nights.","ground_truth":"travel"}'
```

---

## 3) Dual-Mode Classification (`POST /api/v1/emails/classify`)

### Topic Mode (Semantic Classification)

Uses cosine similarity between the email embedding and topic embeddings.

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/emails/classify" \
  -H "Content-Type: application/json" \
  -d '{"subject":"Flight reschedule","body":"Can you change my flight to next Friday?","mode":"topic"}'
```

---

### Email Mode (Retrieval-Based Classification)

Finds the most similar stored email and returns its `ground_truth`.

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/emails/classify" \
  -H "Content-Type: application/json" \
  -d '{"subject":"Change my reservation","body":"I need to modify my hotel booking dates.","mode":"email"}'
```

---

## 4) Extended Feature Pipeline

### Embedding Generator
- Confirmed that the embedding generator combines both subject and body text.

### NonTextCharacterFeatureGenerator
- Implemented a new feature generator that counts non-alphanumeric characters.
- Registered the generator in `FeatureGeneratorFactory`.

Example implementation:

```python
class NonTextCharacterFeatureGenerator(BaseFeatureGenerator):

    def generate_features(self, email: Email) -> Dict[str, Any]:
        import re
        all_text = f"{email.subject} {email.body}"
        non_text_chars = re.findall(r'[^a-zA-Z0-9\s]', all_text)
        return {"non_text_char_count": len(non_text_chars)}

    @property
    def feature_names(self) -> list[str]:
        return ["non_text_char_count"]
```

---

# Demonstration (Screenshots)

### A) Add a New Topic

![Add Topic](screenshots/Adding_a_New_Topic.png)

### B) Classify Using Topic Mode

![Classify Topic Mode](screenshots/Classification_Using_New_Topic.png)

### C) Store Email with Ground Truth

![Store Email](screenshots/Storing_Email_with_Ground_Truth.png)

### D) Classify Using Email Mode

![Classify Email Mode](screenshots/Classification_Using_Stored_Emails.png)

---


# Summary

The system now supports:

- Runtime topic configurability
- Persistent email storage
- Retrieval-based inference
- Dual inference modes in a single endpoint
- Modular feature generation using the factory pattern

This improves extensibility, flexibility, and real-world ML system design.

import requests
import json

# Let's search for human proteins with pathways
url = "https://rest.uniprot.org/uniprotkb/search"
params = {
    "query": "taxonomy_id:9606 AND cc_pathway:*",
    "fields": "accession,cc_subcellular_location,cc_pathway,cc_similarity,cc_domain",
    "format": "json",
    "size": 5
}
resp = requests.get(url, params=params)
if resp.status_code == 200:
    results = resp.json().get("results", [])
    for res in results:
        print(f"Accession: {res.get('primaryAccession')}")
        for c in res.get("comments", []):
            ctype = c.get("commentType")
            if ctype in ["PATHWAY", "SIMILARITY", "DOMAIN"]:
                print(f"Type: {ctype}")
                print(json.dumps(c, indent=2))
        print("=" * 60)
else:
    print(f"Error {resp.status_code}")

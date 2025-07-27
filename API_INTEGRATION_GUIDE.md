# Fashion Image Retrieval API Integration Guide

This guide explains how to integrate the Fashion Image Retrieval API with your frontend application.

## Overview

The API provides a REST endpoint that allows users to upload a reference image and provide modification text to retrieve the top 20 most relevant fashion items from the database.

## API Endpoints

### 1. Health Check
- **URL**: `GET /health`
- **Description**: Check if the API is running
- **Response**: `{"status": "healthy", "message": "API is running"}`

### 2. Get Categories
- **URL**: `GET /categories`
- **Description**: Get available fashion categories
- **Response**: 
```json
{
  "categories": ["dress", "toptee", "shirt"],
  "total_categories": 3
}
```

### 3. Image Retrieval (Main Endpoint)
- **URL**: `POST /retrieve`
- **Description**: Perform composed image retrieval
- **Content-Type**: `application/json`

#### Request Body
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "modification_text": "make it blue",
  "search_categories": ["dress", "shirt"],  // Optional
  "target_color": [0, 0, 255]              // Optional: [R, G, B]
}
```

#### Response
```json
{
  "success": true,
  "results": [
    {
      "rank": 1,
      "image_id": "B002ZG8I60",
      "category": "dress",
      "image_path": "/path/to/image/B002ZG8I60.jpg",
      "color_info": [128, 0, 128],
      "similarity_score": 0.85
    },
    // ... more results (up to 20)
  ],
  "total_results": 20,
  "search_categories": ["dress", "shirt"]
}
```

## Frontend Integration Examples

### JavaScript/HTML Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Fashion Image Retrieval</title>
</head>
<body>
    <div>
        <input type="file" id="imageInput" accept="image/*">
        <input type="text" id="textInput" placeholder="Enter modification text">
        <button onclick="searchImages()">Search</button>
    </div>
    
    <div id="results"></div>

    <script>
        async function searchImages() {
            const imageFile = document.getElementById('imageInput').files[0];
            const modificationText = document.getElementById('textInput').value;
            
            if (!imageFile || !modificationText) {
                alert('Please select an image and enter modification text');
                return;
            }
            
            try {
                const results = await retrieveImages(imageFile, modificationText);
                displayResults(results);
            } catch (error) {
                console.error('Error:', error);
                alert('Error retrieving images');
            }
        }
        
        async function retrieveImages(imageFile, modificationText, searchCategories = null, targetColor = null) {
            // Convert image file to base64
            const base64Image = await fileToBase64(imageFile);
            
            // Prepare request payload
            const payload = {
                image: base64Image,
                modification_text: modificationText
            };
            
            if (searchCategories) {
                payload.search_categories = searchCategories;
            }
            
            if (targetColor) {
                payload.target_color = targetColor;
            }
            
            const response = await fetch('http://localhost:5000/retrieve', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });
            
            if (response.ok) {
                return await response.json();
            } else {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        }
        
        function fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = () => resolve(reader.result);
                reader.onerror = error => reject(error);
            });
        }
        
        function displayResults(results) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';
            
            if (results.success) {
                results.results.forEach(item => {
                    const img = document.createElement('img');
                    img.src = `/images/${item.image_id}.jpg`; // Adjust path as needed
                    img.style.width = '200px';
                    img.style.margin = '10px';
                    
                    const info = document.createElement('div');
                    info.innerHTML = `
                        <p>Rank: ${item.rank}</p>
                        <p>Category: ${item.category}</p>
                        <p>Score: ${item.similarity_score.toFixed(3)}</p>
                    `;
                    
                    resultsDiv.appendChild(img);
                    resultsDiv.appendChild(info);
                });
            }
        }
    </script>
</body>
</html>
```

### React Example

```jsx
import React, { useState } from 'react';

function FashionRetrieval() {
    const [imageFile, setImageFile] = useState(null);
    const [modificationText, setModificationText] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleImageUpload = (event) => {
        setImageFile(event.target.files[0]);
    };

    const handleSearch = async () => {
        if (!imageFile || !modificationText) {
            alert('Please select an image and enter modification text');
            return;
        }

        setLoading(true);
        try {
            const base64Image = await fileToBase64(imageFile);
            
            const response = await fetch('http://localhost:5000/retrieve', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: base64Image,
                    modification_text: modificationText
                })
            });

            if (response.ok) {
                const data = await response.json();
                setResults(data.results);
            } else {
                throw new Error('Search failed');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error retrieving images');
        } finally {
            setLoading(false);
        }
    };

    const fileToBase64 = (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    };

    return (
        <div>
            <div>
                <input type="file" onChange={handleImageUpload} accept="image/*" />
                <input 
                    type="text" 
                    value={modificationText}
                    onChange={(e) => setModificationText(e.target.value)}
                    placeholder="Enter modification text"
                />
                <button onClick={handleSearch} disabled={loading}>
                    {loading ? 'Searching...' : 'Search'}
                </button>
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                {results.map((item) => (
                    <div key={item.rank} style={{ margin: '10px', textAlign: 'center' }}>
                        <img 
                            src={`/images/${item.image_id}.jpg`} 
                            alt={item.image_id}
                            style={{ width: '200px', height: 'auto' }}
                        />
                        <p>Rank: {item.rank}</p>
                        <p>Category: {item.category}</p>
                        <p>Score: {item.similarity_score.toFixed(3)}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default FashionRetrieval;
```

## Setup Instructions

### 1. Start the API Service

```bash
# Navigate to the project directory
cd "composed image retrieval/Fashion_recommender"

# Activate your virtual environment
source venv/bin/activate

# Install API dependencies
pip install -r api_requirements.txt

# Start the API service
python api_service.py
```

The API will start on `http://localhost:5000`

### 2. Test the API

```bash
# Test the client example
python api_client_example.py
```

### 3. Frontend Integration

1. **Image Upload**: Convert the uploaded image to base64 format
2. **API Call**: Send POST request to `/retrieve` endpoint
3. **Display Results**: Show the top 20 results with images and metadata

## Important Notes

### Image Paths
- The API returns `image_path` for each result
- You'll need to serve these images from your frontend server
- Adjust the image paths in your frontend code accordingly

### CORS
- The API has CORS enabled for frontend integration
- If you're running the frontend on a different port, CORS should handle it automatically

### Error Handling
- Always check the `success` field in the response
- Handle network errors and API errors gracefully
- Show appropriate loading states and error messages

### Performance
- The API returns results in ~1-3 seconds depending on the query
- Consider implementing caching for frequently requested images
- The API supports up to 20 results per query

## Advanced Features

### Category Filtering
You can limit the search to specific categories:
```javascript
const results = await retrieveImages(imageFile, modificationText, ['dress', 'shirt']);
```

### Color Specification
You can specify a target color as RGB values:
```javascript
const results = await retrieveImages(imageFile, modificationText, null, [0, 0, 255]); // Blue
```

### Auto Category Detection
If no categories are specified, the API automatically detects the most relevant category based on the modification text.

## Troubleshooting

1. **API not responding**: Check if the service is running on port 5000
2. **CORS errors**: Ensure the API service is running and CORS is enabled
3. **Image not found**: Verify the image paths in your frontend match the API response
4. **Slow responses**: The first request may be slower as models are loaded into memory

## Support

For technical issues or questions about the API integration, refer to the `api_client_example.py` file for working examples. 
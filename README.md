# HNSW-based Vector Storage in Go

This repository provides an in-memory implementation of **HNSW (Hierarchical Navigable Small World)** for high-dimensional vector storage, designed to support efficient approximate nearest neighbor (ANN) search. This implementation is written in Go and can be used to store and search high-dimensional vectors, typically used in machine learning, recommendation systems, and various similarity search applications.

## Features

- **HNSW Indexing**: A memory-efficient, fast, and approximate nearest neighbor search algorithm based on the HNSW graph.
- **In-Memory Storage**: Vectors are stored and queried in memory, making the system fast and responsive.
- **Euclidean Distance**: Uses Euclidean distance for vector similarity computation.
- **Simple API**: Provides easy-to-use functions for adding vectors and querying nearest neighbors.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API](#api)
- [Unit Tests](#unit-tests)
- [License](#license)

## Installation

To get started, you need to have **Go** installed on your system. You can download and install Go from the official website: [https://golang.org/dl/](https://golang.org/dl/).

1. Install dependencies (if any):

```bash
go get github.com/Rustixir/gector
```

2. Build the project (optional if you want to run as a standalone application):

```bash
go build -o hnsw-vector-storage
```

## Usage

This library provides the ability to add vectors and perform nearest neighbor searches. Below is an example of how to use it:

### **Basic Example:**

```go
package main

import (
	"fmt"
	"log"
)

func main() {
	// Initialize the HNSW index with vector dimension 5 and maximum 4 levels
	hnswIndex := NewHNSW(5, 4)

	// Add a vector to the index
	vector1 := generateRandomVector(5)
	hnswIndex.AddVector("vec-1", vector1)

	// Query for nearest neighbors
	query := generateRandomVector(5)
	neighbors := hnswIndex.NearestNeighbors(query, 1)

	// Print the nearest neighbors
	fmt.Println("Nearest Neighbors:", neighbors)
}
```

### **Functions**

- `AddVector(id string, vector Vector)`:
    - Adds a vector to the index.
    - Parameters:
        - `id`: The unique identifier for the vector.
        - `vector`: A `Vector` struct containing the vector's values and ID.

- `NearestNeighbors(query Vector, k int)`:
    - Finds the `k` nearest neighbors of a given query vector.
    - Parameters:
        - `query`: The query vector.
        - `k`: The number of nearest neighbors to retrieve.
    - Returns a list of vectors representing the `k` nearest neighbors.

## API

### **Data Structures**

1. **Vector**:
    - The primary data structure for storing vectors.
    - Fields:
        - `ID`: A unique string identifier for the vector.
        - `Values`: A slice of `float64` representing the vector values.

2. **HNSW**:
    - The main structure responsible for managing the HNSW graph.
    - Methods:
        - `AddVector(id string, vector Vector)`: Adds a vector to the HNSW index.
        - `NearestNeighbors(query Vector, k int) []Vector`: Returns the `k` nearest neighbors to a query vector.

### **Distance Calculation**

The library uses **Euclidean distance** to measure similarity between vectors. The Euclidean distance between two vectors \(A = (a_1, a_2, ..., a_n)\) and \(B = (b_1, b_2, ..., b_n)\) is calculated as:

\[
d(A, B) = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}
\]

### **HNSW Indexing**

- The index is constructed using **HNSW (Hierarchical Navigable Small World)** graphs. The HNSW algorithm is designed for fast approximate nearest neighbor searches in high-dimensional spaces.
- The index is built with multiple levels, where each level contains a subset of vectors and only stores connections to the closest neighbors in the lower levels.

## Unit Tests

Unit tests are provided to ensure that the core functionality of the library works as expected. The tests cover various use cases, including:
- Adding vectors to the index.
- Performing nearest neighbor searches with different numbers of neighbors.
- Handling edge cases like an empty index or querying with a single vector.

### Running Unit Tests

You can run the tests using the Go testing framework:

```bash
go test -v
```

This will execute all the tests and print detailed output. To run specific tests, you can use:

```bash
go test -run TestNearestNeighborsSingleVector
```

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

### **Additional Notes**

- The current implementation is in-memory only. If you want to persist data, consider extending the storage mechanism or integrating with external databases.
- The HNSW algorithm is approximate, so it might not always return the exact nearest neighbors, but it is efficient in high-dimensional spaces.
- The algorithm can be customized by adjusting parameters like vector dimension, number of levels, and number of neighbors.

---

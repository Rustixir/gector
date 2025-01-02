package gector

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// Test for adding vectors to the HNSW index and ensuring they are correctly stored
func TestAddVector(t *testing.T) {
	hnswIndex := NewHNSW(5, 4) // Initialize HNSW index

	// Add a vector
	vector1 := generateRandomVector(5)
	hnswIndex.AddVector("vec-1", vector1)

	// Check if the vector is in the index
	if node, exists := hnswIndex.nodes["vec-1"]; !exists {
		t.Fatalf("Expected vector 'vec-1' to be added to the index, but it wasn't")
	} else {
		if !equalVectors(node.Vector, vector1) {
			t.Errorf("Expected vector 'vec-1' to have the correct vector values, but got %v", node.Vector)
		}
	}
}

// Test for nearest neighbors search with one vector
func TestNearestNeighborsSingleVector(t *testing.T) {
	hnswIndex := NewHNSW(5, 4) // Initialize HNSW index

	// Add a vector
	vector1 := generateRandomVector(5)
	hnswIndex.AddVector("vec-1", vector1)

	// Perform nearest neighbor search
	query := generateRandomVector(5)
	neighbors := hnswIndex.NearestNeighbors(query, 1)

	// Check if the search returns the expected result
	if len(neighbors) != 1 {
		t.Fatalf("Expected 1 nearest neighbor, but got %d", len(neighbors))
	}

	// Check that the nearest neighbor is close to the query
	if euclideanDistance(neighbors[0], query) > euclideanDistance(vector1, query) {
		t.Errorf("Expected the closest neighbor to be 'vec-1', but it wasn't")
	}
}

// Test for NearestNeighbors
func TestNearestNeighbors(t *testing.T) {
	hnswIndex := NewHNSW(5, 4)

	// Add vectors
	vector1 := generateRandomVector(5)
	vector2 := generateRandomVector(5)
	vector3 := generateRandomVector(5)

	hnswIndex.AddVector("vec-1", vector1)
	hnswIndex.AddVector("vec-2", vector2)
	hnswIndex.AddVector("vec-3", vector3)

	// Query vector
	query := generateRandomVector(5)

	// Get nearest neighbors
	neighbors := hnswIndex.NearestNeighbors(query, 2)

	// Ensure we got 2 neighbors
	if len(neighbors) != 2 {
		t.Fatalf("Expected 2 nearest neighbors, but got %d", len(neighbors))
	}

	// Calculate distances to the query for validation
	dist1 := euclideanDistance(query, neighbors[0])
	dist2 := euclideanDistance(query, neighbors[1])

	// Ensure that the first neighbor is closer than the second
	if dist1 > dist2 {
		t.Fatalf("Expected the first neighbor to be closer to the query than the second. Got distances: %f, %f", dist1, dist2)
	}
}

// Test for checking if the HNSW index properly handles edge cases
func TestEdgeCases(t *testing.T) {
	hnswIndex := NewHNSW(5, 4) // Initialize HNSW index

	// Case 1: Query on empty index
	query := generateRandomVector(5)
	neighbors := hnswIndex.NearestNeighbors(query, 3)
	if len(neighbors) != 0 {
		t.Errorf("Expected 0 neighbors for an empty index, but got %d", len(neighbors))
	}

	// Case 2: Single element index, query on single vector
	vector1 := generateRandomVector(5)
	hnswIndex.AddVector("vec-1", vector1)
	neighbors = hnswIndex.NearestNeighbors(query, 1)
	if len(neighbors) != 1 {
		t.Errorf("Expected 1 nearest neighbor, but got %d", len(neighbors))
	}
}

// Test for UpdateVector
func TestUpdateVector(t *testing.T) {
	hnswIndex := NewHNSW(5, 4)

	// Add a vector
	vector1 := generateRandomVector(5)
	hnswIndex.AddVector("vec-1", vector1)

	// Check if the vector exists before update
	if _, exists := hnswIndex.nodes["vec-1"]; !exists {
		t.Errorf("Expected vector with ID 'vec-1' to exist before update")
	}

	// Update the vector with new values
	vector2 := generateRandomVector(5)
	err := hnswIndex.UpdateVector("vec-1", vector2)
	if err != nil {
		t.Errorf("Error updating vector: %v", err)
	}

	// Check if the vector exists after update and its new values
	updatedNode, exists := hnswIndex.nodes["vec-1"]
	if !exists {
		t.Errorf("Expected vector with ID 'vec-1' to exist after update")
	}

	if !equalVectors(updatedNode.Vector, vector2) {
		t.Errorf("Expected updated vector values to match, but they don't")
	}
}

// Test for DeleteVector
func TestDeleteVector(t *testing.T) {
	hnswIndex := NewHNSW(5, 4)

	// Add a vector
	vector1 := generateRandomVector(5)
	hnswIndex.AddVector("vec-1", vector1)

	// Check if the vector exists before delete
	if _, exists := hnswIndex.nodes["vec-1"]; !exists {
		t.Errorf("Expected vector with ID 'vec-1' to exist before delete")
	}

	// Delete the vector
	err := hnswIndex.DeleteVector("vec-1")
	if err != nil {
		t.Errorf("Error deleting vector: %v", err)
	}

	// Check if the vector exists after delete
	if _, exists := hnswIndex.nodes["vec-1"]; exists {
		t.Errorf("Expected vector with ID 'vec-1' to be deleted")
	}
}

// Helper function to compare two vectors
func equalVectors(v1, v2 Vector) bool {
	if len(v1.Values) != len(v2.Values) {
		return false
	}
	for i := 0; i < len(v1.Values); i++ {
		if v1.Values[i] != v2.Values[i] {
			return false
		}
	}
	return true
}

// Helper function to generate random vectors for tests (same as in the main code)
func generateRandomVector(dim int) Vector {
	rand.Seed(time.Now().UnixNano())
	values := make([]float64, dim)
	for i := 0; i < dim; i++ {
		values[i] = rand.Float64() * 100 // Random values between 0 and 100
	}
	return Vector{
		ID:     fmt.Sprintf("vector-%d", rand.Int()),
		Values: values,
	}
}

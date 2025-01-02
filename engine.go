package gector

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// HNSWNode represents a node in the HNSW graph with vector data.
type HNSWNode struct {
	ID        string
	Neighbors []string
	Vector    Vector
}

// HNSW represents the entire HNSW graph.
type HNSW struct {
	// Maps node ID to the actual node
	nodes map[string]*HNSWNode
	// Graph levels: Higher levels have fewer nodes, lower levels more.
	levels []map[string]*HNSWNode
	// Max number of neighbors each node can have
	MaxNeighbors int
	// Maximum number of levels in the graph
	MaxLevels int
}

// NewHNSW creates a new HNSW index.
func NewHNSW(maxNeighbors, maxLevels int) *HNSW {
	return &HNSW{
		nodes:        make(map[string]*HNSWNode),
		levels:       make([]map[string]*HNSWNode, maxLevels),
		MaxNeighbors: maxNeighbors,
		MaxLevels:    maxLevels,
	}
}

// AddVector adds a vector to the HNSW index.
func (hnsw *HNSW) AddVector(id string, vector Vector) {
	// Create a new node with the vector
	node := &HNSWNode{
		ID:     id,
		Vector: vector,
	}

	// Add the node to the bottom level of the graph
	level := hnsw.MaxLevels - 1
	hnsw.addNodeToLevel(node, level)

	// Perform insertion into higher levels based on probability
	for level > 0 && rand.Float64() < 0.5 {
		level--
		hnsw.addNodeToLevel(node, level)
	}

	// Store the node in the map
	hnsw.nodes[id] = node
}

// UpdateVector updates an existing vector with a new one (by deleting the old one and adding the new one)
func (hnsw *HNSW) UpdateVector(id string, newVector Vector) error {
	// Check if the vector exists
	_, exists := hnsw.nodes[id]
	if !exists {
		return fmt.Errorf("vector with id %s not found", id)
	}

	// Remove the old vector (delete node and connections)
	hnsw.DeleteVector(id)

	// Add the new vector with the same ID
	hnsw.AddVector(id, newVector)
	return nil
}

// DeleteVector removes a vector from the HNSW index
func (hnsw *HNSW) DeleteVector(id string) error {
	// Remove the node from each level
	for i := 0; i < hnsw.MaxLevels; i++ {
		delete(hnsw.levels[i], id)
	}
	// Remove the node from the Nodes map
	delete(hnsw.nodes, id)
	return nil
}

// addNodeToLevel adds a node to the specified level.
func (hnsw *HNSW) addNodeToLevel(node *HNSWNode, level int) {
	// Initialize the level map if not yet initialized
	if hnsw.levels[level] == nil {
		hnsw.levels[level] = make(map[string]*HNSWNode)
	}

	// Add the node to the level
	hnsw.levels[level][node.ID] = node

	// Connect the node to its neighbors in this level
	neighbors := hnsw.findNeighbors(node, level)
	node.Neighbors = neighbors
}

// findNeighbors finds the closest neighbors for a node at the specified level.
func (hnsw *HNSW) findNeighbors(node *HNSWNode, level int) []string {
	// Placeholder for nearest neighbor search logic
	// We need to calculate the Euclidean distance and return top K nearest neighbors
	var neighbors []string
	var distances []float64

	// Iterate over nodes in the same level to find the closest ones
	for id, otherNode := range hnsw.levels[level] {
		if node.ID == id {
			continue
		}
		dist := euclideanDistance(node.Vector, otherNode.Vector)
		distances = append(distances, dist)
		neighbors = append(neighbors, id)
	}

	// Sort neighbors by distance
	sort.SliceStable(neighbors, func(i, j int) bool {
		return distances[i] < distances[j]
	})

	// Return the top K neighbors based on MaxNeighbors
	if len(neighbors) > hnsw.MaxNeighbors {
		neighbors = neighbors[:hnsw.MaxNeighbors]
	}

	return neighbors
}

// euclideanDistance calculates the Euclidean distance between two vectors.
func euclideanDistance(v1, v2 Vector) float64 {
	var sum float64
	for i := 0; i < len(v1.Values); i++ {
		diff := v1.Values[i] - v2.Values[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// NearestNeighbors returns the k nearest neighbors to a given query vector
func (hnsw *HNSW) NearestNeighbors(query Vector, k int) []Vector {
	var bestNeighbors []Vector
	var bestDistances []float64

	// Search through all levels and collect the closest neighbors
	for level := hnsw.MaxLevels - 1; level >= 0; level-- {
		var candidates []string
		var distances []float64

		// Iterate through nodes at the current level
		for _, node := range hnsw.levels[level] {
			dist := euclideanDistance(query, node.Vector)
			candidates = append(candidates, node.ID)
			distances = append(distances, dist)
		}

		// Sort neighbors by distance in ascending order
		sort.SliceStable(candidates, func(i, j int) bool {
			return distances[i] < distances[j]
		})

		// Add the best neighbors from this level
		for i := 0; i < k && i < len(candidates); i++ {
			node := hnsw.nodes[candidates[i]]
			bestNeighbors = append(bestNeighbors, node.Vector)
			bestDistances = append(bestDistances, distances[i])
		}
	}

	// Ensure we return only the top k neighbors
	if len(bestNeighbors) > k {
		bestNeighbors = bestNeighbors[:k]
	}

	return bestNeighbors
}

# Coordinate decoding

1. The expression model assigns each original reference cell to one observed spot.
2. The assigned cell count determines the number of physical anchor positions inside that spot.
3. The decoder scores cell-anchor pairs. Hungarian assignment selects one distinct anchor per cell; a Sinkhorn relaxation supplies the training gradient.
4. A bounded residual adjusts each occupied anchor position. Anchor spacing and the available boundary margin determine the displacement limit.
5. The decoder returns one coordinate for each original cell ID.

The occupied-anchor stage freezes the first-stage expression embeddings and cell memberships. Its objective combines local expression-distance relations, global expression-graph relations, neighborhood expression guidance and a normalized residual penalty.

The standard notebook displays an overall coordinate view and a detailed within-spot view. Optional reference coordinates and annotations are loaded only after inference for visualization. They are not used to train the model or match cell identities.

Alternative `original_neural` and `energy` decoding paths remain available through the Python API and workflow configuration.

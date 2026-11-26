"""Exceptions raised by the image store and image loading components."""


class ImageStoreError(Exception):
    """Base exception for image store operations."""


class DuplicateImageArtifactError(ImageStoreError):
    """Raised when an artifact ID is already registered."""


class ImageNotFoundError(ImageStoreError):
    """Raised when an artifact ID does not exist in the store."""


class InvalidImageError(ImageStoreError):
    """Raised when image content is invalid or cannot be decoded."""


class UnsupportedImageFormatError(ImageStoreError):
    """Raised when an image format is not supported."""


class ImageIntegrityError(ImageStoreError):
    """Raised when image bytes fail integrity verification."""


class ImageSizeLimitError(ImageStoreError):
    """Raised when an image exceeds an individual size or dimension limit."""


class ImageStoreCapacityError(ImageStoreError):
    """Raised when the store capacity would be exceeded."""


class ImageArtifactMismatchError(ImageStoreError):
    """Raised when supplied artifact metadata differs from stored metadata."""

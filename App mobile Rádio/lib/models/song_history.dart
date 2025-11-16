class SongHistory {
  final String title;
  final String artist;
  final String? cover;
  final int playedAt; // Unix timestamp

  SongHistory({
    required this.title,
    required this.artist,
    this.cover,
    required this.playedAt,
  });

  factory SongHistory.fromJson(Map<String, dynamic> json) {
    return SongHistory(
      title: json['title'] ?? '',
      artist: json['artist'] ?? '',
      cover: json['cover'],
      playedAt: json['played_at'] ?? 0,
    );
  }

  DateTime get playedAtDateTime => DateTime.fromMillisecondsSinceEpoch(playedAt * 1000);
}

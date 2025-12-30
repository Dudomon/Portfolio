import { memo } from "react";

/**
 * User list showing online participants in a chat room.
 *
 * Presence data comes from Phoenix.Presence on the backend, which uses
 * CRDTs for distributed consistency. The users prop updates via the
 * usePresence hook whenever presence state changes.
 *
 * User status (active/idle) is computed server side based on time since
 * last activity. This keeps client simple and ensures consistent status
 * across all viewers.
 */

interface User {
  id: string;
  username: string;
  onlineAt: number;
  status: string;
}

interface UserListProps {
  users: User[];
  currentUserId: string;
}

export function UserList({ users, currentUserId }: UserListProps): JSX.Element {
  // Sort: current user first, then alphabetically
  const sortedUsers = [...users].sort((a, b) => {
    if (a.id === currentUserId) return -1;
    if (b.id === currentUserId) return 1;
    return a.username.localeCompare(b.username);
  });

  return (
    <div className="user-list">
      <header className="user-list__header">
        <h2 className="user-list__title">Online</h2>
        <span className="user-list__count">{users.length}</span>
      </header>

      <ul className="user-list__items" role="list">
        {sortedUsers.map((user) => (
          <UserItem
            key={user.id}
            user={user}
            isCurrentUser={user.id === currentUserId}
          />
        ))}
      </ul>
    </div>
  );
}

interface UserItemProps {
  user: User;
  isCurrentUser: boolean;
}

const UserItem = memo(function UserItem({
  user,
  isCurrentUser,
}: UserItemProps): JSX.Element {
  return (
    <li className="user-item">
      <span
        className={`user-item__status-dot user-item__status-dot--${user.status}`}
        aria-label={user.status}
      />
      <span className="user-item__name">
        {user.username}
        {isCurrentUser && <span className="user-item__you"> (you)</span>}
      </span>
    </li>
  );
});

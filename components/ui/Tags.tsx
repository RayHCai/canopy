export function Tags({
  tags,
  label,
  className = "",
}: {
  tags: string[];
  label: string;
  className?: string;
}) {
  if (tags.length === 0) return null;
  return (
    <ul aria-label={label} className={`flex flex-wrap gap-1.5 ${className}`}>
      {tags.map((tag) => (
        <li
          key={tag}
          className="inline-flex h-7 items-center rounded-full bg-paper-2 px-3 text-[13px] font-medium text-ink-2"
        >
          {tag}
        </li>
      ))}
    </ul>
  );
}

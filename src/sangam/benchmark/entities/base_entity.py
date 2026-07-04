class BaseEntity:
    _id = 0

    @classmethod
    def generate_id(cls):
        cls._id += 1
        return cls._id

    @property
    def id(self) -> int:
        return self._id

    def __str__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({str(self.to_dict())})"

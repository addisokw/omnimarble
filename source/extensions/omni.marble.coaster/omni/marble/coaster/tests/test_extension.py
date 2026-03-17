"""Basic tests for the marble coaster extension."""

import omni.kit.test


class TestMarbleCoasterExtension(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_extension_loads(self):
        """Verify the extension loaded successfully."""
        import omni.ext
        manager = omni.ext.get_extension_manager()
        self.assertTrue(manager.is_extension_enabled("omni.marble.coaster"))

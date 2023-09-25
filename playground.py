def test_all(self):
    for _ in range(20):
        x, y = random.randrange(2**256), random.randrange(2**256)
        self.assertFalse(ec.is_curve_point((x,y)))
        G_x = ec.multiply(ec.G, x)
        G_y = ec.multiply(ec.G, y)
        self.assertTrue(ec.is_curve_point(G_x))
        self.assertEqual(
            ec.multiply(G_x, y)[0],
            ec.multiply(G_y, x)[0]
        )
        self.assertEqual(
            ec.add(G_x, G_y)[0],
            ec.multiply(ec.G, ec.add_scalar(x, y))[0]
        )
        self.assertEqual(
            ec.substract(G_x, G_y)[0],
            ec.multiply(ec.G, ec.substract_scalar(x, y))[0]
        )
        self.assertEqual(ec.G[0], ec.multiply(ec.divide(ec.G, x), x)[0])